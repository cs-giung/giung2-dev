import os
import sys
sys.path.append('./')

import math
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
from giung2.models.layers import FilterResponseNorm
from giung2.metrics import evaluate_acc, evaluate_nll


class SGHMCState(NamedTuple):
    count: jnp.array
    rng_key: Any
    momentum: Any


class TrainState(train_state.TrainState):
    image_stats: Any

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            gradients   = grads,
            state       = self.opt_state,
            params      = self.params,
            temperature = kwargs.pop('temperature', 1.0))
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs)


def sghmc(learning_rate, seed=0, alpha=0.1):
    """
    Optax implementation of the SGHMC and SGLD.
    
    Args:
        learning_rate : A fixed global scaling factor.
        seed (int) : Seed for the pseudo-random generation process (default: 0).
        alpha (float) : A momentum decay value (default: 0.1)
    """
    def init_fn(params):
        return SGHMCState(
            count    = jnp.zeros([], jnp.int32),
            rng_key  = jax.random.PRNGKey(seed),
            momentum = jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(gradients, state, params=None, temperature=1.0):
        del params
        lr = learning_rate(state.count)
        
        # generate standard gaussian noise
        numvars = len(jax.tree_util.tree_leaves(gradients))
        treedef = jax.tree_util.tree_structure(gradients)
        allkeys = jax.random.split(state.rng_key, num=numvars+1)
        rng_key = allkeys[0]
        noise = jax.tree_util.tree_map(
            lambda p, k: jax.random.normal(k, shape=p.shape),
            gradients, jax.tree_util.tree_unflatten(treedef, allkeys[1:]))

        # compute the dynamics
        momentum = jax.tree_util.tree_map(
            lambda m, g, z: (1 - alpha) * m - lr * g + z * jnp.sqrt(
                2 * alpha * temperature * lr
            ), state.momentum, gradients, noise)
        updates = momentum

        return updates, SGHMCState(
            count    = state.count + 1,
            rng_key  = rng_key,
            momentum = momentum)

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

    if config.model_style == 'FRN-Swish':
        model = _ResNet(
            conv = partial(
                flax.linen.Conv,
                use_bias    = True,
                kernel_init = jax.nn.initializers.he_normal(),
                bias_init   = jax.nn.initializers.zeros),
            norm = FilterResponseNorm,
            relu = flax.linen.swish)

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
    num_epochs_per_cycle = config.num_epochs_quiet + config.num_epochs_noisy

    temperature = optax.join_schedules(
        schedules=[optax.constant_schedule(0.0),] + sum([[
            optax.constant_schedule(1.0),
            optax.constant_schedule(0.0),
        ] for iii in range(1, config.num_cycles + 1)], []),
        boundaries=sum([[
            (iii * num_epochs_per_cycle - config.num_epochs_noisy) * dataloaders['trn_steps_per_epoch'],
            (iii * num_epochs_per_cycle                          ) * dataloaders['trn_steps_per_epoch'],
        ] for iii in range(1, config.num_cycles + 1)], []))
    
    scheduler = optax.join_schedules(
        schedules=[
            optax.cosine_decay_schedule(
                init_value  = config.optim_lr,
                decay_steps = num_epochs_per_cycle * dataloaders['trn_steps_per_epoch'],
            ) for _ in range(1, config.num_cycles + 1)
        ], boundaries=[
            iii * num_epochs_per_cycle * dataloaders['trn_steps_per_epoch']
            for iii in range(1, config.num_cycles + 1)
        ])
    optimizer = sghmc(
        learning_rate = scheduler,
        alpha         = (1.0 - config.optim_momentum))
    
    # build train state
    state = TrainState.create(
        apply_fn      = model.apply,
        params        = variables['params'],
        tx            = optimizer,
        image_stats   = variables['image_stats'])

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler, num_data, temperature):

        # define loss function
        def loss_fn(params):

            # forward pass
            _, new_model_state = state.apply_fn({
                    'params': params,
                    'image_stats': state.image_stats,
                }, batch['images'],
                rngs                = None,
                mutable             = 'intermediates')

            # compute neg_log_likelihood
            logits = new_model_state['intermediates']['cls.logit'][0]                            # [B, K,]
            target = common_utils.onehot(batch['labels'], num_classes=logits.shape[-1])          # [B, K,]
            neg_log_likelihood = -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1) # [B,]
            neg_log_likelihood = num_data * jnp.sum(
                jnp.where(batch['marker'], neg_log_likelihood, jnp.zeros_like(neg_log_likelihood))
            ) / jnp.sum(batch['marker'])

            # compute neg_log_prior
            n_params = sum([p.size for p in jax.tree_util.tree_leaves(params)])
            neg_log_prior = 0.5 * (
                - n_params * jnp.log(2.0 * math.pi)
                + n_params * jnp.log(config.prior_precision + 1e-8)
                + sum([jnp.sum(e**2) for e in jax.tree_util.tree_leaves(params)]) * config.prior_precision)

            # compute posterior_energy
            posterior_energy = neg_log_likelihood + neg_log_prior

            # log metrics
            metrics = OrderedDict({
                'posterior_energy': posterior_energy,
                'neg_log_likelihood': neg_log_likelihood,
                'neg_log_prior': neg_log_prior,
            })
            return posterior_energy, metrics

        # compute losses and gradients
        aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        # get auxiliaries
        metrics = aux[1]
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)
        metrics['temperature'] = temperature(state.step)

        # update train state
        new_state = state.apply_gradients(grads=grads, temperature=temperature(state.step))
        return new_state, metrics

    def step_val(state, batch):

        # forward pass
        _, new_model_state = state.apply_fn({
                'params': state.params,
                'image_stats': state.image_stats,
            }, batch['images'],
            rngs                = None,
            mutable             = 'intermediates')
        
        # compute metrics
        predictions = jax.nn.log_softmax(new_model_state['intermediates']['cls.logit'][0], axis=-1) # [B, K,]
        acc = evaluate_acc(predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics along with softmax predictions
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics, jnp.exp(predictions)

    p_step_trn         = jax.pmap(partial(step_trn,
                                          config=config,
                                          scheduler=scheduler,
                                          num_data=dataloaders['num_data'],
                                          temperature=temperature), axis_name='batch')
    p_step_val         = jax.pmap(        step_val,                 axis_name='batch')
    state              = jax_utils.replicate(state)
    rng                = jax.random.PRNGKey(config.seed)

    # initialize buffer
    val_loader = dataloaders['val_loader'](rng=None)
    val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
    val_marker = jnp.concatenate([batch['marker'].reshape(-1) for batch in val_loader]) # [N,]
    
    val_loader = dataloaders['val_loader'](rng=None)
    val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
    val_labels = jnp.concatenate([batch['labels'].reshape(-1) for batch in val_loader]) # [N,]

    tst_loader = dataloaders['tst_loader'](rng=None)
    tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
    tst_marker = jnp.concatenate([batch['marker'].reshape(-1) for batch in tst_loader]) # [N,]
    
    tst_loader = dataloaders['tst_loader'](rng=None)
    tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
    tst_labels = jnp.concatenate([batch['labels'].reshape(-1) for batch in tst_loader]) # [N,]

    val_ens_predictions = jnp.zeros((val_labels.shape[0], dataloaders['num_classes']))  # [N, K,]
    tst_ens_predictions = jnp.zeros((tst_labels.shape[0], dataloaders['num_classes']))  # [N, K,]

    for cycle_idx, _ in enumerate(range(config.num_cycles), start=1):

        for epoch_idx, _ in enumerate(range(num_epochs_per_cycle), start=1):

            log_str = '[Cycle {:3d}/{:3d}][Epoch {:3d}/{:3d}] '.format(
                cycle_idx, config.num_cycles, epoch_idx, num_epochs_per_cycle)
            rng, data_rng = jax.random.split(rng)

            # ---------------------------------------------------------------------- #
            # Train
            # ---------------------------------------------------------------------- #
            trn_metric = []
            trn_loader = dataloaders['dataloader'](rng=data_rng)
            trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
            for batch_idx, batch in enumerate(trn_loader, start=1):
                state, metrics = p_step_trn(state, batch)
                trn_metric.append(metrics)
            trn_metric = common_utils.get_metrics(trn_metric)
            trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.mean(), trn_metric).items()}
            log_str += ', '.join(f'{k} {v: .3e}' for k, v in trn_summarized.items())

            # ---------------------------------------------------------------------- #
            # Valid
            # ---------------------------------------------------------------------- #
            val_metric = []
            val_predictions = []
            val_loader = dataloaders['val_loader'](rng=None)
            val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
            for batch_idx, batch in enumerate(val_loader, start=1):
                metrics, predictions = p_step_val(state, batch)
                val_metric.append(metrics)
                val_predictions.append(predictions.reshape(-1, predictions.shape[-1]))
            val_metric = common_utils.get_metrics(val_metric)
            val_summarized = {f'val/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), val_metric).items()}
            val_summarized['val/acc'] /= val_summarized['val/cnt']
            val_summarized['val/nll'] /= val_summarized['val/cnt']
            del val_summarized['val/cnt']
            log_str += ', ' + ', '.join(f'{k} {v:.3e}' for k, v in val_summarized.items())

            if epoch_idx % num_epochs_per_cycle == 0:
                val_predictions = jnp.concatenate(val_predictions)
                val_ens_predictions = ((cycle_idx - 1) * val_ens_predictions + val_predictions) / cycle_idx
                acc = evaluate_acc(val_ens_predictions, val_labels, log_input=False, reduction='none')
                nll = evaluate_nll(val_ens_predictions, val_labels, log_input=False, reduction='none')
                acc = jnp.sum(jnp.where(val_marker, acc, jnp.zeros_like(acc))) / jnp.sum(val_marker)
                nll = jnp.sum(jnp.where(val_marker, nll, jnp.zeros_like(nll))) / jnp.sum(val_marker)
                log_str += f', val/ens_acc {acc:.3e}, val/ens_nll {nll:.3e}'

            # ---------------------------------------------------------------------- #
            # Save
            # ---------------------------------------------------------------------- #
            if config.save and epoch_idx % num_epochs_per_cycle == 0:

                tst_metric = []
                tst_predictions = []
                tst_loader = dataloaders['tst_loader'](rng=None)
                tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
                for batch_idx, batch in enumerate(tst_loader, start=1):
                    metrics, predictions = p_step_val(state, batch)
                    tst_metric.append(metrics)
                    tst_predictions.append(predictions.reshape(-1, predictions.shape[-1]))
                tst_metric = common_utils.get_metrics(tst_metric)
                tst_summarized = {f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
                test_acc = tst_summarized['tst/acc'] / tst_summarized['tst/cnt']
                test_nll = tst_summarized['tst/nll'] / tst_summarized['tst/cnt']
                
                tst_predictions = jnp.concatenate(tst_predictions)
                tst_ens_predictions = ((cycle_idx - 1) * tst_ens_predictions + tst_predictions) / cycle_idx
                acc = evaluate_acc(tst_ens_predictions, tst_labels, log_input=False, reduction='none')
                nll = evaluate_nll(tst_ens_predictions, tst_labels, log_input=False, reduction='none')
                acc = jnp.sum(jnp.where(tst_marker, acc, jnp.zeros_like(acc))) / jnp.sum(tst_marker)
                nll = jnp.sum(jnp.where(tst_marker, nll, jnp.zeros_like(nll))) / jnp.sum(tst_marker)
                log_str += ' (test_acc: {:.3e}, test_nll: {:.3e}, test_ens_acc: {:.3e}, test_ens_nll: {:.3e})'.format(
                    test_acc, test_nll, acc, nll)

                _state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
                with gfile.GFile(os.path.join(config.save, f'cycle-{cycle_idx:03d}.ckpt'), 'wb') as fp:
                    fp.write(serialization.to_bytes(_state))
            log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
            print_fn(log_str)

            # wait until computations are done
            jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
            if jnp.isnan(trn_summarized['trn/posterior_energy']):
                break


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_lr', default=1e-7, type=float,
                        help='base learning rate (default: 1e-7)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    
    parser.add_argument('--num_cycles', default=30, type=int,
                        help='the number of cycles for each sample (default: 30)')
    parser.add_argument('--num_epochs_quiet', default=45, type=int,
                        help='the number of epochs for each exploration stage (default: 45)')
    parser.add_argument('--num_epochs_noisy', default=5, type=int,
                        help='the number of epochs for each sampling stage (default: 5)')

    parser.add_argument('--prior_precision', default=1.0, type=float,
                        help='prior precision (default: 1.0)')

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

    launch(args, print_fn)


if __name__ == '__main__':
    main()
