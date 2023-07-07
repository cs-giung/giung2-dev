import os
import sys
sys.path.append('./')

import math
import datetime
import itertools
from typing import Any
from tabulate import tabulate
from itertools import chain
from functools import partial
from collections import OrderedDict

import jax
import jaxlib
import flax
import optax
import jax.numpy as jnp
from flax import jax_utils, serialization
from flax.training import common_utils, train_state
from tensorflow.io.gfile import GFile

from scripts import defaults
from giung2.data import image_processing
from giung2.data.build import load_data, build_dataloader
from giung2.models.resnet import FlaxResNet
from giung2.models.flax.layers import FilterResponseNorm
from giung2.metrics import evaluate_acc, evaluate_nll
from giung2.utils.tree_utils import randn_like_tree


def launch(config, print_fn):

    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
    def prepare_data(batch):
        def _prepare(x):
            return x.reshape(shard_shape + x.shape[1:])
        return jax.tree_util.tree_map(_prepare, batch)

    data = load_data(config.data_root, config.data_name)
    val_transform = image_processing.ToTensorTransform()
    trn_transform = jax.jit(jax.vmap(
        image_processing.TransformChain([
            image_processing.RandomCropTransform(
                size=data.image_shape[1], padding=4),
            image_processing.RandomHFlipTransform(prob=0.5),
            image_processing.ToTensorTransform(),
        ]))) if config.data_augmentation else val_transform
    
    build_trn_loader = lambda rng: jax_utils.prefetch_to_device(
        map(prepare_data, build_dataloader(
            images=data.trn_images,
            labels=data.trn_labels,
            batch_size=config.batch_size,
            rng=rng, shuffle=True, transform=trn_transform,
        )), config.prefetch_factor)
    build_val_loader = lambda rng: jax_utils.prefetch_to_device(
        map(prepare_data, build_dataloader(
            images=data.val_images,
            labels=data.val_labels,
            batch_size=config.batch_size,
            rng=None, shuffle=False, transform=val_transform,
        )), config.prefetch_factor)
    build_tst_loader = lambda rng: jax_utils.prefetch_to_device(
        map(prepare_data, build_dataloader(
            images=data.tst_images,
            labels=data.tst_labels,
            batch_size=config.batch_size,
            rng=None, shuffle=False, transform=val_transform,
        )), config.prefetch_factor)

    trn_steps_per_epoch = math.floor(len(data.trn_images) / config.batch_size)
    NUM_CLASSES = data.num_classes

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    model = FlaxResNet(
        image_size=data.image_shape[1],
        depth=config.resnet_depth,
        widen_factor=config.resnet_width,
        dtype=jnp.float32,
        pixel_mean=(0.49, 0.48, 0.44),
        pixel_std=(0.2, 0.2, 0.2),
        conv = partial(
            flax.linen.Conv,
            use_bias    = True,
            kernel_init = jax.nn.initializers.he_normal(),
            bias_init   = jax.nn.initializers.zeros),
        norm=FilterResponseNorm,
        relu=jax.nn.silu)
        
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(data.image_shape, model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define forward function and specify shapes
    images = next(build_trn_loader(jax.random.PRNGKey(0)))['images']
    output = jax.pmap(model.apply)({
        'params': jax_utils.replicate(variables['params']),
        'image_stats': jax_utils.replicate(variables['image_stats'])}, images)
    FEATURE_DIM = output.shape[-1]

    log_str = f'images.shape: {images.shape}, output.shape: {output.shape}'
    print_fn(log_str)

    # setup trainable parameters
    initial_ext_params = variables['params']
    initial_cls_params = {
        'kernel': jnp.zeros((FEATURE_DIM, NUM_CLASSES)),
        'bias': jnp.zeros((NUM_CLASSES,))}
    params = {'ext': initial_ext_params, 'cls': initial_cls_params}
    log_str = 'The number of trainable parameters: {:d}'.format(
        jax.flatten_util.ravel_pytree(params)[0].size)
    print_fn(log_str)

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler, temperature, noise_rng):

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
            output = model.apply({
                'params': params['ext'],
                'image_stats': state.image_stats}, batch['images'])
            logits = output @ params['cls']['kernel'] + params['cls']['bias']

            # negative_log_likelihood
            smooth = config.optim_label_smoothing
            target = common_utils.onehot(batch['labels'], NUM_CLASSES)
            target = (1.0 - smooth) * target + \
                smooth * jnp.ones_like(target) / NUM_CLASSES
            source = jax.nn.log_softmax(logits, axis=-1)
            negative_log_likelihood = -jnp.sum(target * source, axis=-1)
            negative_log_likelihood = jnp.mean(negative_log_likelihood)
            negative_log_likelihood *= len(data.trn_images)

            # negative_log_prior
            negative_log_prior = 0.0

            xs = jax.tree_util.tree_leaves(params['ext'])
            negative_log_prior += 0.5 * sum([
                jnp.sum(jnp.square(x)) for x in xs
            ]) / config.prior_var
            
            xs = jax.tree_util.tree_leaves(params['cls'])
            negative_log_prior += 0.5 * sum([
                jnp.sum(jnp.square(x)) for x in xs
            ]) / config.prior_var

            # posterior_energy
            posterior_energy = negative_log_likelihood + negative_log_prior

            # log metrics
            metrics = OrderedDict({
                'posterior_energy': posterior_energy,
                'negative_log_likelihood': negative_log_likelihood,
                'negative_log_prior': negative_log_prior})
            
            posterior_energy /= len(data.trn_images)
            return posterior_energy, metrics

        # compute losses and gradients
        aux, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # compute norms of weights and gradients
        w_norm = _global_norm(state.params)
        g_norm = _global_norm(grads)
        if config.optim_global_clipping:
            grads = _clip_by_global_norm(grads, g_norm)

        # inject noise to gradients
        lr = scheduler(state.step)
        sqrt_lr = jnp.sqrt(lr)
        step_size = lr / len(data.trn_images)
        temperature = temperature(state.step)
        alpha = config.optim_momentum
        noise = randn_like_tree(noise_rng, grads)
        grads = jax.tree_util.tree_map(
            lambda g, z: sqrt_lr * g + z * jnp.sqrt(
                2.0 * temperature * (1.0 - alpha) / len(data.trn_images)
            ), grads, noise)

        # get auxiliaries
        metrics = jax.lax.pmean(aux[1], axis_name='batch')
        metrics['w_norm'] = w_norm
        metrics['g_norm'] = g_norm
        metrics['lr'] = lr
        metrics['step_size'] = step_size
        metrics['temperature'] = temperature

        # update train state
        new_state = state.apply_gradients(grads=grads)
        noise_rng = jax.random.split(noise_rng)[1]
        return new_state, metrics, noise_rng
    
    # define optimizer with scheduler
    num_epochs = config.num_epochs_quiet + config.num_epochs_noisy
    num_cycles = config.num_cycles_burnin + config.num_cycles_sample

    temperature = optax.join_schedules(
        schedules=[
            optax.constant_schedule(0.0),
            optax.constant_schedule(config.posterior_tempering),] * num_cycles,
        boundaries=list(chain.from_iterable([[
            num_epochs*trn_steps_per_epoch*i \
                + config.num_epochs_quiet*trn_steps_per_epoch,
            num_epochs*trn_steps_per_epoch*i \
                + config.num_epochs_quiet*trn_steps_per_epoch \
                + config.num_epochs_noisy*trn_steps_per_epoch,
            ] for i in range(num_cycles)]))[:-1])
    
    scheduler = optax.join_schedules(
        schedules=[
            optax.cosine_decay_schedule(
                init_value  = config.optim_lr,
                decay_steps = num_epochs * trn_steps_per_epoch),] * num_cycles,
        boundaries=list(chain.from_iterable([[
            num_epochs*trn_steps_per_epoch*i \
                + config.num_epochs_quiet*trn_steps_per_epoch \
                + config.num_epochs_noisy*trn_steps_per_epoch,
            ] for i in range(num_cycles)]))[:-1])
    sqrt_scheduler = lambda e: jnp.sqrt(scheduler(e))
    optimizer = optax.sgd(sqrt_scheduler, momentum=config.optim_momentum)

    # build and replicate train state
    class TrainState(train_state.TrainState):
        image_stats: Any = None

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
        image_stats=variables['image_stats'])
    state = jax_utils.replicate(state)

    # run optimization
    def apply_fn(images, state):
        return model.apply({
            'params': state.params['ext'],
            'image_stats': state.image_stats}, images
        ) @ state.params['cls']['kernel'] + state.params['cls']['bias']
    p_apply_fn = jax.pmap(apply_fn)
    
    data_rng = jax.random.PRNGKey(config.seed)
    noise_rng = jax_utils.replicate(jax.random.PRNGKey(config.seed))
    p_step_trn = jax.pmap(partial(
        step_trn, config=config,
        scheduler=scheduler, temperature=temperature), axis_name='batch')

    val_ens_preds = jnp.zeros((len(data.val_labels), NUM_CLASSES))
    tst_ens_preds = jnp.zeros((len(data.tst_labels), NUM_CLASSES))
    for cycle_idx in itertools.count(start=1):
            
        # rendezvous
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if cycle_idx == num_cycles + 1:
            break

        for epoch_idx in itertools.count(start=1):
            
            # rendezvous
            jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
            if epoch_idx == num_epochs + 1:
                break

            # --------------------------------------------------------------- #
            # Train
            # --------------------------------------------------------------- #
            log_str = '[Cycle {:3d}/{:3d}][Epoch {:3d}/{:3d}] '.format(
                cycle_idx, num_cycles, epoch_idx, num_epochs)
            trn_summarized, ind_summarized, ens_summarized = {}, {}, {}
            trn_metric = []

            data_rng = jax.random.split(data_rng)[1]
            trn_iter = build_trn_loader(data_rng)
            for batch_idx, batch in enumerate(trn_iter, start=1):
                if jnp.sum(batch['marker']) < config.batch_size:
                    break
                state, metrics, noise_rng = \
                    p_step_trn(state, batch, noise_rng=noise_rng)
                trn_metric.append(metrics)
                if batch_idx == trn_steps_per_epoch:
                    break

            trn_metric = common_utils.get_metrics(trn_metric)
            trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
                lambda e: e.mean(), trn_metric).items()}

            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())

            # --------------------------------------------------------------- #
            # Individual
            # --------------------------------------------------------------- #
            val_iter = build_val_loader(None)
            val_ind_preds = jax.nn.softmax(jnp.concatenate([
                p_apply_fn(batch['images'], state).reshape(-1, NUM_CLASSES)
                for batch in val_iter])[:data.val_labels.shape[0]], axis=-1)
            ind_summarized['val/acc'] = evaluate_acc(
                val_ind_preds, data.val_labels, log_input=False)
            ind_summarized['val/nll'] = evaluate_nll(
                val_ind_preds, data.val_labels, log_input=False)

            tst_iter = build_tst_loader(None)
            tst_ind_preds = jax.nn.softmax(jnp.concatenate([
                p_apply_fn(batch['images'], state).reshape(-1, NUM_CLASSES)
                for batch in tst_iter])[:data.tst_labels.shape[0]], axis=-1)
            ind_summarized['tst/acc'] = evaluate_acc(
                tst_ind_preds, data.tst_labels, log_input=False)
            ind_summarized['tst/nll'] = evaluate_nll(
                tst_ind_preds, data.tst_labels, log_input=False)
            
            log_str += ', '
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in ind_summarized.items())
            
            # --------------------------------------------------------------- #
            # Cumulative
            # --------------------------------------------------------------- #
            if cycle_idx > config.num_cycles_burnin and epoch_idx == num_epochs:
                val_ens_preds = (val_ind_preds + (
                    cycle_idx - config.num_cycles_burnin - 1) * val_ens_preds
                ) / (cycle_idx - config.num_cycles_burnin)
                ens_summarized['val/ens_acc'] = evaluate_acc(
                    val_ens_preds, data.val_labels, log_input=False)
                ens_summarized['val/ens_nll'] = evaluate_nll(
                    val_ens_preds, data.val_labels, log_input=False)

                tst_ens_preds = (tst_ind_preds + (
                    cycle_idx - config.num_cycles_burnin - 1) * tst_ens_preds
                ) / (cycle_idx - config.num_cycles_burnin)
                ens_summarized['tst/ens_acc'] = evaluate_acc(
                    tst_ens_preds, data.tst_labels, log_input=False)
                ens_summarized['tst/ens_nll'] = evaluate_nll(
                    tst_ens_preds, data.tst_labels, log_input=False)
                
                log_str += ', '
                log_str += ', '.join(
                    f'{k} {v:.3e}' for k, v in ens_summarized.items())

            # --------------------------------------------------------------- #
            # Save
            # --------------------------------------------------------------- #
            if config.save and epoch_idx == num_epochs:
                sample_ckpt = {
                    'params': state.params,
                    'image_stats': state.image_stats}
                sample_ckpt = jax.device_get(
                    jax.tree_util.tree_map(lambda x: x[0], sample_ckpt))
                
                sample_path = os.path.join(
                    config.save, f'cycle_{cycle_idx:03d}.ckpt')
                with GFile(sample_path, 'wb') as fp:
                    fp.write(serialization.to_bytes(sample_ckpt))
            
            # logging current iteration
            print_fn(log_str)

            # terminate training if posterior_energy is nan
            if jnp.isnan(trn_summarized['trn/posterior_energy']):
                break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument(
        '--num_epochs_quiet', default=0, type=int,
        help='the number of quiet epochs for each cycle (default: 0)')
    parser.add_argument(
        '--num_epochs_noisy', default=50, type=int,
        help='the number of noisy epochs for each cycle (default: 50)')
    
    parser.add_argument(
        '--num_cycles_burnin', default=20, type=int,
        help='the number of cycles for burn-in (default: 20)')
    parser.add_argument(
        '--num_cycles_sample', default=30, type=int,
        help='the number of cycles for sampling (default: 30)')

    parser.add_argument(
        '--optim_lr', default=0.1, type=float,
        help='base step size (default: 0.1)')
    parser.add_argument(
        '--optim_momentum', default=0.9, type=float,
        help='momentum coefficient (default: 0.9)')

    parser.add_argument(
        '--optim_label_smoothing', default=0.0, type=float,
        help='label smoothing regularization (default: 0.0)')
    parser.add_argument(
        '--optim_global_clipping', default=None, type=float,
        help='global norm for the gradient clipping (default: None)')

    parser.add_argument(
        '--data_augmentation', default=False, type=defaults.str2bool,
        help='it specifies whether augmentation is applied (default: False)')
    parser.add_argument(
        '--posterior_tempering', default=1.0, type=float,
        help='temperature value for posterior tempering (default: 1.0)')
    parser.add_argument(
        '--prior_var', default=0.2, type=float,
        help='variance value for all trainable parameters (default: 0.2)')

    parser.add_argument(
        '--save', default=None, type=str,
        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for training (default: None)')

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
