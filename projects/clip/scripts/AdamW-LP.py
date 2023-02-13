import os
import sys
sys.path.append('./')

import copy
import math
import einops
import datetime
from tqdm import tqdm
from tabulate import tabulate
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jaxlib
import optax
import flax
from flax import jax_utils, serialization
from flax.core import unfreeze
from flax.training import common_utils, train_state
from tensorflow.io import gfile

import torchvision
from transformers import FlaxCLIPModel, CLIPTokenizer
from timm.data.transforms_factory import transforms_imagenet_train

from scripts import defaults
from giung2.data import imagenet
from giung2.metrics import evaluate_acc, evaluate_nll


TrainState = train_state.TrainState


def launch(config, print_fn):
    
    # setup input-transformations
    preprocess_trn = transforms_imagenet_train(
        img_size = 224,
        mean     = (0.48145466, 0.45782750, 0.40821073),
        std      = (0.26862954, 0.26130258, 0.27577711))
    preprocess_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        lambda e: e.convert('RGB'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.48145466, 0.45782750, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )])

    # build model
    model = FlaxCLIPModel.from_pretrained(config.clip_name)

    # ['logit_scale', 'text_model', 'text_projection', 'vision_model', 'visual_projection']
    pretrained_params = copy.deepcopy(model.params)

    ########################################
    # Zeroshot Classification
    ########################################
    log_str = 'Setup zeroshot classification...'
    log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    tokenizer        = CLIPTokenizer.from_pretrained(config.clip_name)
    get_txt_features = jax.jit(model.get_text_features)
    zeroshot_weights = []
    for classname in tqdm(imagenet.openai_classnames):
        texts = tokenizer([t(classname) for t in imagenet.openai_imagenet_template], padding=True, return_tensors='jax')
        feats = get_txt_features(
            input_ids      = texts.input_ids,
            attention_mask = texts.attention_mask,
            params         = pretrained_params)
        feats = feats / jnp.sqrt(jnp.sum(feats**2, axis=-1, keepdims=True))
        feats = jnp.mean(feats, axis=0, keepdims=True)
        feats = feats / jnp.sqrt(jnp.sum(feats**2, axis=-1, keepdims=True))
        zeroshot_weights.append(feats)
    zeroshot_weights = jnp.transpose(
        jnp.concatenate(zeroshot_weights), (1, 0)
    ) * jnp.exp(pretrained_params['logit_scale'])

    log_str = 'Run zeroshot classification...'
    log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    def predict(images):
        f_vecs = model.get_image_features(images, params=pretrained_params)
        f_vecs = f_vecs / jnp.sqrt(jnp.sum(f_vecs**2, axis=-1, keepdims=True))
        logits = f_vecs @ zeroshot_weights
        return logits
    predict = jax.pmap(predict, axis_name='batch')
    shard   = lambda x: einops.rearrange(x, '(d b) ... -> d b ...', d=jax.local_device_count())
    unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

    for dataset_name in ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']:
        dataset = getattr(imagenet, dataset_name)(
            preprocess_val, location=config.data_root, batch_size=config.batch_size, num_workers=config.num_workers)
        dataloader = dataset.test_loader
        
        ACC, NLL, CNT = 0.0, 0.0, 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = jnp.array(batch['images'])
            logits = unshard(
                predict(shard(jnp.concatenate([
                    images, jnp.zeros([config.batch_size - images.shape[0], *images.shape[1:]], images.dtype)
                ]))))[:images.shape[0]]

            project_logits = getattr(dataset, 'project_logits', None)
            if project_logits is not None:
                logits = project_logits(logits)
            
            labels = batch['labels']
            if hasattr(dataset, 'project_labels'):
                labels = dataset.project_labels(labels)
            labels = jnp.array(labels)

            ACC += evaluate_acc(jax.nn.log_softmax(logits, axis=-1), labels, log_input=True, reduction='sum')
            NLL += evaluate_nll(jax.nn.log_softmax(logits, axis=-1), labels, log_input=True, reduction='sum')
            CNT += images.shape[0]
        print_fn('- {:14s} : ACC={:.4f}, NLL={:.4f}, CNT={:d}'.format(dataset_name, ACC / CNT, NLL / CNT, CNT))

    ########################################
    # Optimization
    ########################################
    def apply_fn(images, params):
        f_vecs = model.get_image_features(images, params=pretrained_params)
        f_vecs = f_vecs / jnp.sqrt(jnp.sum(f_vecs**2, axis=-1, keepdims=True))
        return f_vecs @ params['cls']

    def step_trn(state, images, labels, scheduler):

        # define loss function
        def loss_fn(params):
            logits = apply_fn(images, params)
            target = common_utils.onehot(labels, num_classes=1000)
            loss = -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = jnp.mean(loss)
            return loss

        # define gradient clipping function
        def _global_norm(updates):
            return jnp.sqrt(
                sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(updates)]))

        def _clip_by_global_norm(updates):
            global_norm = _global_norm(updates)
            return jax.tree_util.tree_map(
                lambda e: jnp.where(global_norm < config.optim_clipping, e, (e / global_norm) * config.optim_clipping), updates)

        # compute losses and gradients
        aux, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # clip gradients
        _global_norm_before_clipping = _global_norm(grads)
        if config.optim_clipping:
            grads = _clip_by_global_norm(grads)
        
        # update state
        new_state = state.apply_gradients(grads=grads)

        # get metrics
        loss = jax.lax.pmean(aux, axis_name='batch')
        metrics = OrderedDict({
            'loss': loss,
            'norm': _global_norm_before_clipping,
            'lr': scheduler(state.step),
        })
        return new_state, metrics

    def step_val(state, images, labels):
        logits = apply_fn(images, state.params)
        acc = evaluate_acc(jax.nn.log_softmax(logits, axis=-1)[:labels.shape[0]], labels, log_input=True, reduction='sum')
        nll = evaluate_nll(jax.nn.log_softmax(logits, axis=-1)[:labels.shape[0]], labels, log_input=True, reduction='sum')
        metrics = OrderedDict({'acc': acc, 'nll': nll, 'cnt': labels.shape[0]})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    # define dataset
    trn_loader = getattr(imagenet, 'ImageNet')(
        preprocess_trn, location=config.data_root, batch_size=config.batch_size, num_workers=config.num_workers).train_loader
    val_loader = getattr(imagenet, 'ImageNet')(
        preprocess_val, location=config.data_root, batch_size=config.batch_size, num_workers=config.num_workers).test_loader

    # define optimizer and scheduler
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value       = 0.0,
                end_value        = config.optim_lr,
                transition_steps = math.floor(0.1 * config.optim_ne * len(trn_loader)),
            ),
            optax.cosine_decay_schedule(
                init_value       = config.optim_lr,
                decay_steps      = math.floor(0.9 * config.optim_ne * len(trn_loader)),
            )
        ], boundaries=[math.floor(0.1 * config.optim_ne * len(trn_loader)),])
    optimizer = optax.adamw(
        scheduler, b1=config.optim_b1, b2=config.optim_b2,
        eps=config.optim_eps, weight_decay=config.optim_weight_decay)

    # define train state
    state = train_state.TrainState.create(
        apply_fn = apply_fn,
        tx       = optimizer,
        params   = {
            'cls': jnp.zeros_like(zeroshot_weights) if config.clip_zero_head else zeroshot_weights})

    # setup optimization
    state      = jax_utils.replicate(state)
    p_step_trn = jax.pmap(partial(step_trn, scheduler=scheduler), axis_name='batch')
    p_step_val = jax.pmap(step_val, axis_name='batch')
    shard      = lambda x: einops.rearrange(x, '(d b) ... -> d b ...', d=jax.local_device_count())
    unshard    = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

    # start optimization
    for epoch_idx, _ in enumerate(range(config.optim_ne), start=1):
        log_str = '[Epoch {:3d}/{:3d}] '.format(epoch_idx, config.optim_ne)

        # train epoch
        trn_metrics = []
        for batch_idx, batch in enumerate(trn_loader, start=1):
            images, labels = jnp.array(batch['images']), jnp.array(batch['labels'])
            state, metrics = p_step_trn(state, shard(images), shard(labels))
            trn_metrics.append(metrics)
        trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
            lambda e: e.mean(), common_utils.get_metrics(trn_metrics)).items()}

        # valid epoch
        val_metrics = []
        for batch_idx, batch in enumerate(val_loader, start=1):
            images, labels = jnp.array(batch['images']), jnp.array(batch['labels'])
            images = jnp.concatenate([images, jnp.zeros([config.batch_size - images.shape[0], *images.shape[1:]], images.dtype)])
            metrics = p_step_val(state, shard(images), shard(labels))
            val_metrics.append(metrics)
        val_summarized = {f'val/{k}': v for k, v in jax.tree_util.tree_map(
            lambda e: e.mean(), common_utils.get_metrics(val_metrics)).items()}
        val_summarized['val/acc'] /= val_summarized['val/cnt']
        val_summarized['val/nll'] /= val_summarized['val/cnt']
        val_summarized.pop('val/cnt')

        # log optimization
        summarized = {**trn_summarized, **val_summarized}
        log_str += ', '.join(f'{k} {v:.3e}' for k, v in summarized.items())
        log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

        # save
        if config.save:
            _state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
            with gfile.GFile(os.path.join(config.save, 'checkpoint.ckpt'), 'wb') as fp:
                fp.write(serialization.to_bytes(_state))

        # rendezvous
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    # evaluate
    log_str = 'Evaluate the final state...'
    log_str = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    _state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    def predict(images):
        f_vecs = model.get_image_features(images, params=pretrained_params)
        f_vecs = f_vecs / jnp.sqrt(jnp.sum(f_vecs**2, axis=-1, keepdims=True))
        logits = f_vecs @ _state.params['cls']
        return logits
    predict = jax.pmap(predict, axis_name='batch')
    shard   = lambda x: einops.rearrange(x, '(d b) ... -> d b ...', d=jax.local_device_count())
    unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

    for dataset_name in ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ImageNetA', 'ImageNetSketch']:
        dataset = getattr(imagenet, dataset_name)(
            preprocess_val, location=config.data_root, batch_size=config.batch_size, num_workers=config.num_workers)
        dataloader = dataset.test_loader
        
        ACC, NLL, CNT = 0.0, 0.0, 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = jnp.array(batch['images'])
            logits = unshard(
                predict(shard(jnp.concatenate([
                    images, jnp.zeros([config.batch_size - images.shape[0], *images.shape[1:]], images.dtype)
                ]))))[:images.shape[0]]

            project_logits = getattr(dataset, 'project_logits', None)
            if project_logits is not None:
                logits = project_logits(logits)
            
            labels = batch['labels']
            if hasattr(dataset, 'project_labels'):
                labels = dataset.project_labels(labels)
            labels = jnp.array(labels)

            ACC += evaluate_acc(jax.nn.log_softmax(logits, axis=-1), labels, log_input=True, reduction='sum')
            NLL += evaluate_nll(jax.nn.log_softmax(logits, axis=-1), labels, log_input=True, reduction='sum')
            CNT += images.shape[0]
        print_fn('- {:14s} : ACC={:.4f}, NLL={:.4f}, CNT={:d}'.format(dataset_name, ACC / CNT, NLL / CNT, CNT))


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument('--optim_ne', default=10, type=int,
                        help='the number of training epochs (default: 10)')
    parser.add_argument('--optim_lr', default=1e-7, type=float,
                        help='base learning rate (default: 1e-7)')
    parser.add_argument('--optim_b1', default=0.9, type=float,
                        help='rate for the first moment of past gradients (default: 0.9)')
    parser.add_argument('--optim_b2', default=0.999, type=float,
                        help='rate for the second moment of past gradients (default: 0.999)')
    parser.add_argument('--optim_eps', default=1e-08, type=float,
                        help='it applied to denominator outside of the square root (default: 1e-08)')
    parser.add_argument('--optim_weight_decay', default=0.0001, type=float,
                        help='strength of the weight decay regularization (default: 0.0001)')
    parser.add_argument('--optim_clipping', default=None, type=float,
                        help='global norm for the gradient clipping (default: None)')

    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')

    args = parser.parse_args()
    
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
