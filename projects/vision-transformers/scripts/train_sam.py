import os
import sys
sys.path.append('./')

import math
import datetime
import itertools
import numpy as np
from tabulate import tabulate
from functools import partial
from collections import OrderedDict

import jax
import jaxlib
import flax
import optax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from flax import jax_utils, serialization
from flax.training import common_utils, train_state
from tensorflow.io.gfile import GFile

from scripts import defaults
from giung2.data.tfds import input_pipeline
from transformers import ViTConfig, FlaxViTModel
from giung2.metrics import evaluate_acc, evaluate_nll


def launch(config, print_fn):

    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
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
    trn_iter = map(prepare_tf_data, input_pipeline.create_trn_split(
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
    vit_config = ViTConfig(
        hidden_size=384, num_attention_heads=6, intermediate_size=1536)
    
    model = FlaxViTModel(
        config=vit_config, input_shape=(1, 224, 224, 3),
        seed=config.seed, dtype=jnp.dtype(config.dtype))

    # define forward function and specify shapes
    pixel_mean = jnp.array([
        0.48145466, 0.45782750, 0.40821073]).reshape(1, 3, 1, 1)
    pixel_std = jnp.array([
        0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
    
    def get_features(images, params):
        images = (images.transpose(0, 3, 1, 2) - pixel_mean) / pixel_std
        return model(images, params).pooler_output

    images = next(trn_iter)['images']
    output = jax.pmap(get_features)(images, jax_utils.replicate(model.params))
    FEATURE_DIM = output.shape[-1]

    log_str = f'images.shape: {images.shape}, output.shape: {output.shape}'
    print_fn(log_str)

    # setup trainable parameters
    initial_ext_params = model.params
    initial_cls_params = jnp.zeros((FEATURE_DIM, NUM_CLASSES))

    params = {'ext': initial_ext_params, 'cls': initial_cls_params}
    log_str = 'The number of trainable parameters: {:d}'.format(
        jax.flatten_util.ravel_pytree(params)[0].size)
    print_fn(log_str)

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler):

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

            # get features
            output = get_features(batch['images'] / 255.0, params['ext'])
            logits = output @ params['cls']

            # loss
            smooth = config.optim_label_smoothing
            target = common_utils.onehot(
                batch['labels'], NUM_CLASSES) * (1.0 - smooth) + smooth
            log_p = jax.nn.log_sigmoid(logits)
            log_not_p = jax.nn.log_sigmoid(-logits)
            loss = -jnp.sum(
                target * log_p + (1.0 - target) * log_not_p, axis=-1)
            loss = jnp.mean(loss)

            # log metrics
            metrics = OrderedDict({'loss': loss})
            return loss, metrics

        # define grad function
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # get noisy parameters
        (_, aux), grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, axis_name='batch')
        norm = _global_norm(grad)
        grad = jax.tree_util.tree_map(lambda g: g / norm, grad)
        noisy_params = jax.tree_util.tree_map(
            lambda p, g: p + config.optim_sam_rho * g, state.params, grad)

        # compute losses and gradients
        aux, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(noisy_params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # compute norms of weights and gradients
        w_norm = _global_norm(state.params)
        g_norm = _global_norm(grads)
        if config.optim_global_clipping:
            grads = _clip_by_global_norm(grads, g_norm)

        # get auxiliaries
        metrics = jax.lax.pmean(aux[1], axis_name='batch')
        metrics['w_norm'] = w_norm
        metrics['g_norm'] = g_norm
        metrics['lr'] = scheduler(state.step)

        # update train state
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics
    
    # define optimizer with scheduler
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value       = 0.0,
                end_value        = config.optim_lr,
                transition_steps = math.floor(0.1 * config.optim_ni)),
            optax.cosine_decay_schedule(
                init_value       = config.optim_lr,
                decay_steps      = math.floor(0.9 * config.optim_ni))
        ], boundaries=[
            math.floor(0.1 * config.optim_ni),
        ])
    optimizer = optax.adamw(
        scheduler, b1=config.optim_b1, b2=config.optim_b2,
        eps=config.optim_eps, eps_root=config.optim_eps_root,
        mu_dtype=jnp.dtype(config.dtype),
        weight_decay=config.optim_weight_decay)

    # build and replicate train state
    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=params, tx=optimizer)
    state = jax_utils.replicate(state)

    def apply_fn(images, state):
        return get_features(images, state.params['ext']) @ state.params['cls']
    p_apply_fn = jax.pmap(apply_fn)

    # run optimization
    best_acc = 0.0
    p_step_trn = jax.pmap(partial(
        step_trn, config=config, scheduler=scheduler), axis_name='batch')

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
        state, metrics = p_step_trn(state, batch)
        trn_metric.append(metrics)

        if iter_idx % 1000 == 0:
            trn_summarized, val_summarized, tst_summarized = {}, {}, {}
            
            trn_metric = common_utils.get_metrics(trn_metric)
            trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
                lambda e: e.mean(), trn_metric).items()}
            trn_metric = []

            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())

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
            val_summarized['val/best_acc'] = max(
                val_summarized['val/acc'], best_acc)

            log_str += ', '
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in val_summarized.items())

            # --------------------------------------------------------------- #
            # Save
            # --------------------------------------------------------------- #
            if best_acc < val_summarized['val/acc']:

                log_str += ' (best_acc: {:.3e} -> {:.3e})'.format(
                    best_acc, val_summarized['val/acc'])
                best_acc = val_summarized['val/acc']

                best_state = state
                best_state = jax.device_get(
                    jax.tree_util.tree_map(lambda x: x[0], best_state))

                if config.save:
                    best_path = os.path.join(config.save, 'best_state.ckpt')
                    with GFile(best_path, 'wb') as fp:
                        fp.write(serialization.to_bytes(best_state))
                
            # logging current iteration
            print_fn(log_str)

            # terminate training if loss is nan
            if jnp.isnan(trn_summarized['trn/loss']):
                break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument(
        '--optim_ni', default=100000, type=int,
        help='the number of training iterations (default: 100000)')
    parser.add_argument(
        '--optim_lr', default=3e-03, type=float,
        help='base learning rate (default: 3e-03)')
    parser.add_argument(
        '--optim_b1', default=0.9, type=float,
        help='rate for the first moment of past gradients (default: 0.9)')
    parser.add_argument(
        '--optim_b2', default=0.999, type=float,
        help='rate for the second moment of past gradients (default: 0.999)')
    parser.add_argument(
        '--optim_eps', default=1e-08, type=float,
        help='epsilon value outside of the square root (default: 1e-08)')
    parser.add_argument(
        '--optim_eps_root', default=0.0, type=float,
        help='epsilon value inside of the square root (default: 0.0)')
    parser.add_argument(
        '--optim_weight_decay', default=0.3, type=float,
        help='weight decay coefficient (default: 0.3)')
    parser.add_argument(
        '--optim_sam_rho', default=0.1, type=float,
        help='sam rho (default: 0.1)')

    parser.add_argument(
        '--optim_label_smoothing', default=0.0001, type=float,
        help='label smoothing regularization (default: 0.0001)')
    parser.add_argument(
        '--optim_global_clipping', default=1.0, type=float,
        help='global norm for the gradient clipping (default: 1.0)')

    parser.add_argument(
        '--save', default=None, type=str,
        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for training (default: None)')

    parser.add_argument(
        '--dtype', default='float32', type=str,
        help='dtype of computation and accumulator (default: float32)')

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
