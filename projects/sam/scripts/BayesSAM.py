import os
import sys
sys.path.append('./')

import datetime
from tabulate import tabulate
from functools import partial
from typing import Any, NamedTuple
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jaxlib
import optax
import flax
from flax import jax_utils, serialization
from flax.training import common_utils, train_state
from tensorflow.io import gfile

from scripts import defaults
from giung2.data.build import build_dataloaders
from giung2.models.resnet import FlaxResNet
from giung2.metrics import evaluate_acc, evaluate_nll

from optax._src import numerics
from optax._src.utils import canonicalize_dtype, cast_tree


class TrainState(train_state.TrainState):
    image_stats: Any
    batch_stats: Any

    def apply_gradients(self, **kwargs):
        updates, new_opt_state = self.tx.update(
            gradients_z = kwargs.pop('gradients_z', None),
            gradients_e = kwargs.pop('gradients_e', None),
            state       = self.opt_state,
            params      = self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step      = self.step + 1,
            params    = new_params,
            opt_state = new_opt_state,
            **kwargs)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target, keys_tree)


def bsam(learning_rate, b1=0.9, b2=0.999, eps=0.1,
         mu_dtype=None, weight_decay=0.001) -> optax.GradientTransformation:
    
    mu_dtype = canonicalize_dtype(mu_dtype)
    
    def init_fn(params):
        mu = jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.ones_like, params)
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(gradients_z, gradients_e, state, params):
        lr = learning_rate(state.count)
        mu = jax.tree_util.tree_map(
            lambda g, p, t: (1.0 - b1) * (g + weight_decay * p) + b1 * t,
            gradients_e, params, state.mu)
        nu = jax.tree_util.tree_map(
            lambda g, p, t: (1.0 - b2) * (jnp.sqrt(t) * jnp.abs(g) + weight_decay + eps) + b2 * t,
            gradients_z, params, state.nu)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = optax.bias_correction(mu, b1, count_inc)
        nu_hat = optax.bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(lambda m, v: -lr * m / v, mu_hat, nu_hat)
        mu = cast_tree(mu, mu_dtype)
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
    
    return optax.GradientTransformation(init_fn, update_fn)


def launch(config, print_fn):
    
    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices()[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    _ResNet = partial(
        FlaxResNet,
        depth        = config.model_depth,
        widen_factor = config.model_width,
        dtype        = model_dtype,
        pixel_mean   = defaults.PIXEL_MEAN,
        pixel_std    = defaults.PIXEL_STD,
        num_classes  = dataloaders['num_classes'])

    if config.model_style == 'BN-ReLU':
        model = _ResNet()

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define dynamic_scale
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        raise NotImplementedError('fp16 training on GPU is currently not available...')

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value  = config.optim_lr,
        decay_steps = config.optim_ne * dataloaders['trn_steps_per_epoch'])
    optimizer = bsam(
        learning_rate=scheduler, b1=config.optim_b1, b2=config.optim_b2,
        eps=config.optim_eps, weight_decay=config.optim_weight_decay)
    
    # build train state
    state = TrainState.create(
        apply_fn      = model.apply,
        params        = variables['params'],
        tx            = optimizer,
        image_stats   = variables['image_stats'],
        batch_stats   = variables['batch_stats'])

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler, num_data, bsam_key):

        _, new_bsam_key = jax.random.split(bsam_key)

        # define loss function
        def loss_fn(params):

            # forward pass
            _, new_model_state = state.apply_fn({
                    'params': params,
                    'image_stats': state.image_stats,
                    'batch_stats': state.batch_stats,
                }, batch['images'],
                rngs                = None,
                mutable             = ['intermediates', 'batch_stats'],
                use_running_average = False)

            # compute loss
            logits = new_model_state['intermediates']['cls.logit'][0]                   # [B, K,]
            target = common_utils.onehot(batch['labels'], num_classes=logits.shape[-1]) # [B, K,]
            loss = -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)      # [B,]
            loss = jnp.sum(
                jnp.where(batch['marker'], loss, jnp.zeros_like(loss))
            ) / jnp.sum(batch['marker'])

            # log metrics
            metrics = OrderedDict({'loss': loss})
            return loss, (metrics, new_model_state)

        # forward pass through the network to compute batch_stats and metrics...
        _, (metrics, new_model_state) = loss_fn(state.params)
        
        # define grad function
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # get gaussian-noisy parameters
        scale = state.opt_state.nu
        noise = tree_random_normal_like(bsam_key, state.params)
        noisy_params_z = jax.tree_util.tree_map(
            lambda p, z, s: p + jnp.sqrt(
                1.0 / (num_data * s)
            ) * z, state.params, noise, scale)

        # get adversarial-noisy parameters
        _, gradients_z = grad_fn(noisy_params_z)
        gradients_z = jax.lax.pmean(gradients_z, axis_name='batch')
        gradients_z = jax.tree_util.tree_map(lambda g, s: g / s, gradients_z, scale)
        noisy_params_e = jax.tree_util.tree_map(
            lambda p, g: p + config.rho * g, state.params, gradients_z)

        # compute losses and gradients at the adversarial-noisy_params
        _, gradients_e = grad_fn(noisy_params_e)
        gradients_e = jax.lax.pmean(gradients_e, axis_name='batch')
        
        # log metrics
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)
        metrics['scale'] = jnp.mean(jax.flatten_util.ravel_pytree(scale)[0])

        # update train state
        new_state = state.apply_gradients(
            gradients_z=gradients_z,
            gradients_e=gradients_e,
            batch_stats=new_model_state['batch_stats'])

        return new_state, metrics, new_bsam_key

    def step_val(state, batch):

        # forward pass
        _, new_model_state = state.apply_fn({
                'params': state.params,
                'image_stats': state.image_stats,
                'batch_stats': state.batch_stats,
            }, batch['images'],
            rngs                = None,
            mutable             = 'intermediates',
            use_running_average = True)
        
        # compute metrics
        predictions = jax.nn.log_softmax(new_model_state['intermediates']['cls.logit'][0], axis=-1) # [B, K,]
        acc = evaluate_acc(predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_trn         = jax.pmap(partial(step_trn, config=config, scheduler=scheduler,
                                          num_data=dataloaders['num_data']*config.num_data_factor), axis_name='batch')
    p_step_val         = jax.pmap(        step_val,                                                 axis_name='batch')
    state              = jax_utils.replicate(state)
    rng                = jax.random.PRNGKey(config.seed)
    bsam_key           = common_utils.shard_prng_key(rng)
    best_acc           = 0.0
    test_acc           = 0.0
    test_nll           = float('inf')

    for epoch_idx, _ in enumerate(range(config.optim_ne), start=1):

        log_str = '[Epoch {:5d}/{:5d}] '.format(epoch_idx, config.optim_ne)
        rng, data_rng = jax.random.split(rng)

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=data_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader, start=1):
            state, metrics, bsam_key = p_step_trn(state, batch, bsam_key=bsam_key)
            trn_metric.append(metrics)
        trn_metric = common_utils.get_metrics(trn_metric)
        trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.mean(), trn_metric).items()}
        log_str += ', '.join(f'{k} {v:.3e}' for k, v in trn_summarized.items())
        
        # synchronize batch normalization statistics
        state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

        # ---------------------------------------------------------------------- #
        # Valid
        # ---------------------------------------------------------------------- #
        val_metric = []
        val_loader = dataloaders['val_loader'](rng=None)
        val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        for batch_idx, batch in enumerate(val_loader, start=1):
            metrics = p_step_val(state, batch)
            val_metric.append(metrics)
        val_metric = common_utils.get_metrics(val_metric)
        val_summarized = {f'val/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), val_metric).items()}
        val_summarized['val/acc'] /= val_summarized['val/cnt']
        val_summarized['val/nll'] /= val_summarized['val/cnt']
        del val_summarized['val/cnt']
        log_str += ', ' + ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

        # ---------------------------------------------------------------------- #
        # Save
        # ---------------------------------------------------------------------- #
        if config.save and best_acc < val_summarized['val/acc']:

            tst_metric = []
            tst_loader = dataloaders['tst_loader'](rng=None)
            tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
            for batch_idx, batch in enumerate(tst_loader, start=1):
                metrics = p_step_val(state, batch)
                tst_metric.append(metrics)
            tst_metric = common_utils.get_metrics(tst_metric)
            tst_summarized = {f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
            test_acc = tst_summarized['tst/acc'] / tst_summarized['tst/cnt']
            test_nll = tst_summarized['tst/nll'] / tst_summarized['tst/cnt']
            
            log_str += ' (best_acc: {:.3e} -> {:.3e}, test_acc: {:.3e}, test_nll: {:.3e})'.format(
                best_acc, val_summarized['val/acc'], test_acc, test_nll)
            best_acc = val_summarized['val/acc']

            _state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
            with gfile.GFile(os.path.join(config.save, 'best_acc.ckpt'), 'wb') as fp:
                fp.write(serialization.to_bytes(_state))
        log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_ne', default=200, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=0.1, type=float,
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--optim_b1', default=0.9, type=float,
                        help='rate for the first moment of past gradients (default: 0.9)')
    parser.add_argument('--optim_b2', default=0.999, type=float,
                        help='rate for the second moment of past gradients (default: 0.999)')
    parser.add_argument('--optim_eps', default=0.1, type=float,
                        help='it applied to denominator outside of the square root (default: 0.1)')
    parser.add_argument('--optim_weight_decay', default=0.001, type=float,
                        help='weight decay coefficient (default: 0.001)')
    
    parser.add_argument('--rho', default=1.0, type=float,
                        help='radius value (default: 1.0)')
    parser.add_argument('--num_data_factor', default=1.0, type=float,
                        help='the number of train data (default: 1.0)')

    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])

    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )
    
    if args.save is not None:
        args.save = os.path.abspath(args.save)
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    print_fn = partial(print, flush=True)
    if args.save:
        def print_fn(s):
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
                fp.write(s + '\n')
            print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__ + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__ + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = f'Multiple local devices are detected:\n{jax.local_devices()}\n'
        log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
