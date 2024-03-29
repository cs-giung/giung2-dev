import os
import sys
sys.path.append('./')

import math
import datetime
import itertools
import numpy as np
from typing import Any
from tabulate import tabulate
from functools import partial
from collections import OrderedDict

import jax
import jaxlib
import flax
import optax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils, serialization
from flax.training import common_utils, train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from tensorflow.io.gfile import GFile

from scripts import defaults
from giung2.data import image_processing
from giung2.data.tfds import input_pipeline
from giung2.models.resnet import FlaxResNet
from giung2.metrics import evaluate_acc, evaluate_nll


def launch(config, print_fn):

    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # setup mixed precision training if specified
    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
        model_dtype = jnp.float16
    elif config.mixed_precision and platform == 'tpu':
        dynamic_scale = None
        model_dtype = jnp.bfloat16
    else:
        dynamic_scale = None
        model_dtype = jnp.float32

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
    from giung2.data.tfds.input_pipeline import _random_crop, _random_flip

    def create_trn_split_simclr(data_builder, batch_size, split='train',
                                dtype=tf.float32, image_size=224, cache=True):
        data = data_builder.as_dataset(
            split=split, shuffle_files=True,
            decoders={'image': tfds.decode.SkipDecoding()})
        image_decoder = data_builder.info.features['image'].decode_example
        shuffle_buffer_size = min(
            16*batch_size, data_builder.info.splits[split].num_examples)
        def decode_example(example):
            image = image_decoder(example['image'])
            jmage = _random_flip(_random_crop(image, image_size))
            jmage = tf.reshape(jmage, [image_size, image_size, 3])
            jmage = tf.cast(jmage, dtype=dtype)
            kmage = _random_flip(_random_crop(image, image_size))
            kmage = tf.reshape(kmage, [image_size, image_size, 3])
            kmage = tf.cast(kmage, dtype=dtype)
            image = tf.stack([jmage, kmage], axis=0)
            return {'images': image, 'labels': example['label']}
        if cache:
            data = data.cache()
        data = data.repeat()
        data = data.shuffle(shuffle_buffer_size)
        data = data.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.batch(batch_size, drop_remainder=True)
        data = data.prefetch(tf.data.AUTOTUNE)
        return data

    def prepare_tf_data(batch):
        batch['images'] = batch['images']._numpy()
        batch['labels'] = batch['labels']._numpy()
        batch['marker'] = np.ones_like(batch['labels'])
        def _prepare(x):
            if x.shape[0] < config.batch_size:
                x = np.concatenate([x, np.zeros([
                    config.batch_size - x.shape[0], *x.shape[1:]
                ], x.dtype)])
            return x.reshape(shard_shape + x.shape[1:])
        return jax.tree_util.tree_map(_prepare, batch)

    dataset_builder = tfds.builder(config.data_name)

    trn_split = 'train'
    trn_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[trn_split].num_examples / config.batch_size)
    trn_iter = map(prepare_tf_data, create_trn_split_simclr(
        dataset_builder, config.batch_size, split=trn_split))
    trn_iter = jax_utils.prefetch_to_device(trn_iter, config.prefetch_factor)

    val_split = 'validation'
    val_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[val_split].num_examples / config.batch_size)
    val_iter = map(prepare_tf_data, input_pipeline.create_val_split(
        dataset_builder, config.batch_size, split=val_split))
    val_iter = jax_utils.prefetch_to_device(val_iter, config.prefetch_factor)

    NUM_CLASSES = 1000

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    model = FlaxResNet(
        image_size=224,
        depth=config.resnet_depth,
        widen_factor=config.resnet_width,
        dtype=model_dtype,
        pixel_mean=(0.48145466, 0.45782750, 0.40821073),
        pixel_std=(0.26862954, 0.26130258, 0.27577711),
        norm=partial(flax.linen.BatchNorm,
                     momentum=0.9, epsilon=1e-5, axis_name='batch'))
        
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones((1, 224, 224, 3), model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define forward function and specify shapes
    images = next(trn_iter)['images'][:, 0]
    output = jax.pmap(model.apply)({
        'params': jax_utils.replicate(variables['params']),
        'batch_stats': jax_utils.replicate(variables['batch_stats']),
        'image_stats': jax_utils.replicate(variables['image_stats'])}, images)
    FEATURE_DIM = output.shape[-1]

    log_str = f'images.shape: {images.shape}, output.shape: {output.shape}'
    print_fn(log_str)

    # setup trainable parameters
    PROJECT_DIM = config.projection_dim

    params = {
        'ext': variables['params'],
        'cls': jnp.zeros((FEATURE_DIM, NUM_CLASSES)),
        'proj_head_w0': jax.random.normal(
            jax.random.PRNGKey(config.seed + 1), (FEATURE_DIM, FEATURE_DIM)
        ) / FEATURE_DIM**0.5,
        'proj_head_s0': jnp.ones((FEATURE_DIM,)),
        'proj_head_b0': jnp.zeros((FEATURE_DIM,)),
        'proj_head_w1': jax.random.normal(
            jax.random.PRNGKey(config.seed + 2), (FEATURE_DIM, PROJECT_DIM)
        ) / FEATURE_DIM**0.5,
        'proj_head_s1': jnp.ones((PROJECT_DIM,)),
        'proj_head_b1': jnp.zeros((PROJECT_DIM,))}
    log_str = 'The number of trainable parameters: {:d}'.format(
        jax.flatten_util.ravel_pytree(params)[0].size)
    print_fn(log_str)

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #
    augment_image = jax.jit(jax.vmap(image_processing.TransformChain([
        image_processing.TransformChain([
            image_processing.RandomBrightnessTransform(0.2, 1.8),
            image_processing.RandomContrastTransform(0.2, 1.8),
            image_processing.RandomSaturationTransform(0.2, 1.8),
            image_processing.RandomHueTransform(0.2),
        ], prob=0.8),
        image_processing.RandomGrayscaleTransform(prob=0.2),
        image_processing.TransformChain([
            image_processing.RandomGaussianBlurTransform(
                kernel_size=int(0.1 * 224), sigma=(0.1, 2.0)),
        ], prob=0.5)])))
    
    def step_trn(state, batch, config, scheduler, dynamic_scale, data_rng):

        def _global_norm(updates):
            return jnp.sqrt(sum([jnp.sum(jnp.square(e))
                                 for e in jax.tree_util.tree_leaves(updates)]))
        
        def _clip_by_global_norm(updates, global_norm):
            return jax.tree_util.tree_map(
                lambda e: jnp.where(
                    global_norm < config.optim_global_clipping, e,
                    (e / global_norm) * config.optim_global_clipping), updates)
        
        # define loss function
        def loss_fn(params):

            # obtain two views of the same example
            rngs = jax.random.split(data_rng)
            images = jnp.stack([
                augment_image(jax.random.split(
                    rngs[0], batch['images'][:, 0].shape[0]), batch['images'][:, 0]),
                augment_image(jax.random.split(
                    rngs[1], batch['images'][:, 1].shape[0]), batch['images'][:, 1]),
                ], axis=1)
            images = images.reshape((images.shape[0] * 2, images.shape[2],
                                     images.shape[3], images.shape[4]))

            # forward a base encoder network
            output, new_model_state = model.apply({
                'params': params['ext'],
                'batch_stats': state.batch_stats,
                'image_stats': state.image_stats}, images / 255.0,
                mutable='batch_stats', use_running_average=False)
            
            # negative_log_likelihood (for online linear evaluation)
            logits = jax.lax.stop_gradient(output) @ params['cls']
            labels = jnp.stack([batch['labels'], batch['labels']], axis=1)
            labels = labels.reshape((labels.shape[0] * 2))
            negative_log_likelihood = -jnp.sum(
                common_utils.onehot(labels, logits.shape[1]) \
                    * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            negative_log_likelihood = jnp.mean(negative_log_likelihood)

            # forward a projection head
            output = output @ params['proj_head_w0']
            output = jax.lax.rsqrt(
                jnp.var(output, axis=-1, keepdims=True) + 1e-05
            ) * (output - jnp.mean(output, axis=-1, keepdims=True))
            output *= params['proj_head_s0'].reshape(1, FEATURE_DIM)
            output += params['proj_head_b0'].reshape(1, FEATURE_DIM)
            output = jax.nn.relu(output)

            output = output @ params['proj_head_w1']
            output = jax.lax.rsqrt(
                jnp.var(output, axis=-1, keepdims=True) + 1e-05
            ) * (output - jnp.mean(output, axis=-1, keepdims=True))
            output *= params['proj_head_s1'].reshape(1, PROJECT_DIM)
            output += params['proj_head_b1'].reshape(1, PROJECT_DIM)

            output = output / (
                jnp.linalg.norm(output, axis=-1, keepdims=True) + 1e-10)

            # contrastive_loss
            simmat = output @ output.T
            mask = 1 - np.eye(output.shape[0], dtype=int)
            pos_idx = (
                np.arange(output.shape[0]), 2 * np.repeat(
                    np.arange(output.shape[0] // 2)[:, np.newaxis, ...], 2, 1
                ).reshape(-1))
            neg_mask = np.ones(
                (output.shape[0], output.shape[0] - 1), dtype=int)
            neg_mask[pos_idx] = 0

            simmat = simmat[mask == 1].reshape(simmat.shape[0], -1)
            pos = simmat[pos_idx][:, jnp.newaxis]
            neg = simmat[neg_mask == 1].reshape(simmat.shape[0], -1)

            logits = jnp.concatenate((pos, neg), axis=1) / config.temperature
            labels = jnp.zeros((logits.shape[0],), dtype=int)
            contrastive_loss = -jnp.sum(
                common_utils.onehot(labels, logits.shape[1]) \
                    * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            contrastive_loss = jnp.mean(contrastive_loss)

            # loss
            loss = contrastive_loss + negative_log_likelihood

            # log metrics
            metrics = OrderedDict({
                'loss': loss,
                'contrastive_loss': contrastive_loss,
                'negative_log_likelihood': negative_log_likelihood})
            return loss, (metrics, new_model_state)

        # compute losses and gradients
        if dynamic_scale:
            dynamic_scale, is_fin, aux, grads = dynamic_scale.value_and_grad(
                loss_fn, has_aux=True, axis_name='batch')(state.params)
        else:
            aux, grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
        
        # compute norms of weights and gradients
        w_norm = _global_norm(state.params)
        g_norm = _global_norm(grads)
        if config.optim_global_clipping:
            grads = _clip_by_global_norm(grads, g_norm)

        # get auxiliaries
        metrics = jax.lax.pmean(aux[1][0], axis_name='batch')
        metrics['w_norm'] = w_norm
        metrics['g_norm'] = g_norm
        metrics['lr'] = scheduler(state.step)

        # update train state
        new_state = state.apply_gradients(
            grads=grads, batch_stats=aux[1][1]['batch_stats'])
        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin),
                    new_state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin),
                    new_state.params, state.params))
            metrics['dyn_scale'] = dynamic_scale.scale
        
        new_data_rng = jax.random.split(data_rng)[1]
        return new_state, metrics, new_data_rng
    
    # define optimizer with scheduler
    sqrt_lr = config.optim_lr * math.sqrt(config.batch_size)
    warm_up = min(10 * trn_steps_per_epoch, math.floor(0.1 * config.optim_ni))
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=sqrt_lr, transition_steps=warm_up),
            optax.cosine_decay_schedule(
                init_value=sqrt_lr, decay_steps=(config.optim_ni - warm_up)),
        ], boundaries=[warm_up,])
    optimizer = optax.lars(
        scheduler, weight_decay=config.optim_weight_decay,
        weight_decay_mask=jax.tree_util.tree_map(lambda e: e.ndim > 1, params),
        trust_ratio_mask=jax.tree_util.tree_map(lambda e: e.ndim > 1, params),
        momentum=config.optim_momentum)
    
    # build and replicate train state
    class TrainState(train_state.TrainState):
        batch_stats: Any = None
        image_stats: Any = None

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
        batch_stats=variables['batch_stats'],
        image_stats=variables['image_stats'])
    state = jax_utils.replicate(state)

    def apply_fn(images, state):
        return model.apply({
            'params': state.params['ext'],
            'batch_stats': state.batch_stats,
            'image_stats': state.image_stats,
        }, images, use_running_average=True) @ state.params['cls']
    p_apply_fn = jax.pmap(apply_fn)

    # run optimization
    data_rng = jax.random.split(
        jax.random.PRNGKey(config.seed), local_device_count)
    p_step_trn = jax.pmap(partial(
        step_trn, config=config, scheduler=scheduler), axis_name='batch')
    sync_batch_stats = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    
    if dynamic_scale:
        dynamic_scale = jax_utils.replicate(dynamic_scale)

    trn_metric = []
    for iter_idx in itertools.count(start=1):
        
        # rendezvous
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

        # terminate training
        if iter_idx == config.optim_ni + 1:
            break

        # ------------------------------------------------------------------- #
        # Train
        # ------------------------------------------------------------------- #
        log_str = '[Iter {:7d}/{:7d}] '.format(iter_idx, config.optim_ni)

        batch = next(trn_iter)
        state, metrics, data_rng = p_step_trn(
            state, batch, dynamic_scale=dynamic_scale, data_rng=data_rng)
        trn_metric.append(metrics)

        if iter_idx % 1000 == 0:
            trn_summarized, val_summarized = {}, {}
            
            trn_metric = common_utils.get_metrics(trn_metric)
            trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
                lambda e: e.mean(), trn_metric).items()}
            trn_metric = []

            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())

            # synchronize batch_stats across replicas
            state = state.replace(
                batch_stats=sync_batch_stats(state.batch_stats))

            # --------------------------------------------------------------- #
            # Valid
            # --------------------------------------------------------------- #
            acc, nll, cnt = 0.0, 0.0, 0
            for batch_idx, batch in enumerate(val_iter, start=1):
                logits = p_apply_fn(batch['images'] / 255.0, state)
                logits = logits.reshape(-1, NUM_CLASSES)
                labels = batch['labels'].reshape(-1)
                marker = batch['marker'].reshape(-1)
                pre = jax.nn.log_softmax(logits, axis=-1)
                acc += jnp.sum(jnp.where(marker, evaluate_acc(
                    pre, labels, log_input=True, reduction='none'
                ), marker))
                nll += jnp.sum(jnp.where(marker, evaluate_nll(
                    pre, labels, log_input=True, reduction='none'
                ), marker))
                cnt += jnp.sum(marker)
                if batch_idx == val_steps_per_epoch:
                    break
            val_summarized['val/acc'] = acc / cnt
            val_summarized['val/nll'] = nll / cnt

            log_str += ', '
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in val_summarized.items())
            
            # --------------------------------------------------------------- #
            # Save
            # --------------------------------------------------------------- #
            save_ckpt = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'image_stats': state.image_stats}
            save_ckpt = jax.device_get(
                jax.tree_util.tree_map(lambda x: x[0], save_ckpt))

            if config.save:
                save_path = os.path.join(config.save, 'checkpoint.ckpt')
                with GFile(save_path, 'wb') as fp:
                    fp.write(serialization.to_bytes(save_ckpt))
                
            # logging current iteration
            print_fn(log_str)

            # terminate training if loss is nan
            if jnp.isnan(trn_summarized['trn/loss']):
                break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument(
        '--projection_dim', default=128, type=int,
        help='projection output dimensionality (default: 128)')
    parser.add_argument(
        '--temperature', default=0.1, type=float,
        help='temperature parameter in the NT-Xent loss (default: 0.1)')

    parser.add_argument(
        '--optim_ni', default=32000, type=int,
        help='the number of training iterations (default: 32000)')
    parser.add_argument(
        '--optim_lr', default=0.075, type=float,
        help='square root learning rate scaling will be used (default: 0.075)')
    parser.add_argument(
        '--optim_momentum', default=0.9, type=float,
        help='momentum coefficient (default: 0.9)')
    parser.add_argument(
        '--optim_weight_decay', default=1e-06, type=float,
        help='weight decay coefficient (default: 1e-06)')

    parser.add_argument(
        '--optim_label_smoothing', default=0.0, type=float,
        help='label smoothing regularization (default: 0.0)')
    parser.add_argument(
        '--optim_global_clipping', default=None, type=float,
        help='global norm for the gradient clipping (default: None)')

    parser.add_argument(
        '--save', default=None, type=str,
        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for training (default: None)')

    parser.add_argument(
        '--mixed_precision', default=False, type=defaults.str2bool,
        help='run mixed precision training if specified (default: False)')

    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big'))

    if args.save is not None:
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    def print_fn(s):
        s = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + s
        if args.save is not None:
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
                fp.write(s + '\n')
        print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__
            + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__
            + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__
            + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__
            + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = (
            'Multiple local devices are detected:\n'
            f'{jax.local_devices()}\n')
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
