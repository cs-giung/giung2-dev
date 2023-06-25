import math
import numpy as np
from collections import namedtuple

import jax
import jax.numpy as jnp
import datasets


def load_data(data_name, tokenizer, max_length=None, truncation=True):

    actual_task = 'mnli' if data_name == 'mnli-mm' else data_name
    num_labels = 3 if actual_task == 'mnli' \
        else 1 if data_name == 'stsb' else 2
    sentence_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentece', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }[actual_task]

    data = namedtuple('data', [
        'trn_input_ids', 'trn_attention_mask', 'trn_labels', 'trn_num_rows',
        'val_input_ids', 'val_attention_mask', 'val_labels', 'val_num_rows',
        'tst_input_ids', 'tst_attention_mask', 'tst_labels', 'tst_num_rows',
        'num_labels'])
    
    def preprocess_fn(examples):
        texts = (
            (examples[sentence_keys[0]],) if sentence_keys[1] is None else
            (examples[sentence_keys[0]], examples[sentence_keys[1]]))
        processed = tokenizer(
            *texts, padding='max_length',
            max_length=max_length, truncation=truncation)
        processed['labels'] = examples['label']
        return processed

    dset = datasets.load_dataset('glue', actual_task)
    dset = dset.map(
        preprocess_fn, batched=True, remove_columns=dset['train'].column_names)
    
    return data(
        np.array(dset['train'].input_ids),
        np.array(dset['train'].attention_mask),
        np.array(dset['train'].labels),
        dset['train'].num_rows,
        np.array(dset['validation'].input_ids),
        np.array(dset['validation'].attention_mask),
        np.array(dset['validation'].labels),
        dset['validation'].num_rows,
        np.array(dset['test'].input_ids),
        np.array(dset['test'].attention_mask),
        np.array(dset['test'].labels),
        dset['test'].num_rows,
        num_labels)


def build_dataloader(input_ids, attention_mask, labels,
                     batch_size, rng=None, shuffle=False):
    
    # shuffle the entire dataset if specified
    num_rows = len(input_ids)
    _shuffled = jax.random.permutation(rng, num_rows) \
        if shuffle else jnp.arange(num_rows)
    
    input_ids = input_ids[_shuffled]
    attention_mask = attention_mask[_shuffled]
    labels = labels[_shuffled]

    # add padding to process the entire dataset
    marker = np.ones(num_rows, dtype=bool)
    num_batches = math.ceil(num_rows / batch_size)
    
    padded_input_ids = np.concatenate([
        input_ids, np.zeros([
            num_batches*batch_size - len(input_ids),
            *input_ids.shape[1:]], input_ids.dtype)])
    padded_attention_mask = np.concatenate([
        attention_mask, np.zeros([
            num_batches*batch_size - len(attention_mask),
            *attention_mask.shape[1:]], attention_mask.dtype)])
    padded_labels = np.concatenate([
        labels, np.zeros([
            num_batches*batch_size - len(labels),
            *labels.shape[1:]], labels.dtype)])
    padded_marker = np.concatenate([
        marker, np.zeros([
            num_batches*batch_size - len(marker),
            *marker.shape[1:]], marker.dtype)])
    
    # define generator using yield
    batch_indices = jnp.arange(len(padded_input_ids))
    batch_indices = batch_indices.reshape((num_batches, batch_size))
    for batch_idx in batch_indices:
        batch = {
            'input_ids': jnp.array(padded_input_ids[batch_idx]),
            'attention_mask': jnp.array(padded_attention_mask[batch_idx]),
            'labels': jnp.array(padded_labels[batch_idx]),
            'marker': jnp.array(padded_marker[batch_idx])}
        yield batch
