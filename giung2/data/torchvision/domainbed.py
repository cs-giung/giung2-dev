"""
Code is adapted from https://github.com/facebookresearch/DomainBed, which is
available under the following MIT license.

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import hashlib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import transforms
from timm.data import ImageDataset
from timm.data.transforms import ToNumpy, str_to_interp_mode


_TRANSFORMS_FACTORY = {
    'eval': transforms.Compose([
        transforms.Resize(
            (224, 224), interpolation=str_to_interp_mode('bicubic')),
        ToNumpy()]),
    'preset0': transforms.Compose([
        transforms.Resize(
            (224, 224), interpolation=str_to_interp_mode('bicubic')),
        ToNumpy()]),
    'preset1': transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.7, 1.0),
            interpolation=str_to_interp_mode('bicubic')),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(p=0.1),
        ToNumpy()]),
    }


def seed_hash(*args):
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


class MultipleDomainDataset:
    N_STEPS = 5001         # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8          # Default, subclasses may override
    ENVIRONMENTS = None    # Subclasses should override
    INPUT_SHAPE = None     # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentImageDataset(MultipleDomainDataset):

    def __init__(self, root, preset='preset1'):
        super().__init__()
        self.environments = [f.name for f in os.scandir(root) if f.is_dir()]
        self.environments = sorted(self.environments)

        self.trn_transform = _TRANSFORMS_FACTORY[preset]
        self.val_transform = _TRANSFORMS_FACTORY['eval']
        
        self.datasets = []
        for environment in self.environments:
            path = os.path.join(root, environment)
            env_dataset = ImageDataset(path)
            self.datasets.append(env_dataset)


class VLCS(MultipleEnvironmentImageDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    CLASSNAMES = ['bird', 'car', 'chair', 'dog', 'person']

    def __init__(self, root):
        self.dir = os.path.join(root, 'domainbed/VLCS/')
        super().__init__(self.dir)


class PACS(MultipleEnvironmentImageDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['art_painting', 'cartoon', 'photo', 'sketch']
    CLASSNAMES = [
        'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __init__(self, root):
        self.dir = os.path.join(root, 'domainbed/PACS/')
        super().__init__(self.dir)


class OfficeHome(MultipleEnvironmentImageDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['Art', 'Clipart', 'Product', 'Real-World']
    CLASSNAMES = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle',
        'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards',
        'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
        'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
        'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives',
        'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug',
        'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
        'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator',
        'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
        'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush',
        'Toys', 'Trash_Can', 'Webcam']
    
    def __init__(self, root):
        self.dir = os.path.join(root, 'domainbed/office_home/')
        super().__init__(self.dir)


class TerraIncognita(MultipleEnvironmentImageDataset):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = [
        'location_100', 'location_38', 'location_43', 'location_46']
    CLASSNAMES = [
        'bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit',
        'raccoon', 'squirrel']

    def __init__(self, root):
        self.dir = os.path.join(root, 'domainbed/terra_incognita/')
        super().__init__(self.dir)


class SplitDataset(torch.utils.data.Dataset):

    def __init__(self, underlying_dataset, keys):
        super(SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

    def __getitem__(self, key):
        images, labels = self.underlying_dataset[self.keys[key]]
        ret = {'labels': labels}
        for key, transform in self.transforms.items():
            ret[key] = transform(images)
        return ret

    def __len__(self):
        return len(self.keys)


class InfiniteSampler(torch.utils.data.Sampler):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:

    def __init__(self, dataset, batch_size, num_workers, prefetch_factor):
        super().__init__()

        sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True)

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                batch_sampler=InfiniteSampler(batch_sampler)))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:

    def __init__(self, dataset, batch_size, num_workers, prefetch_factor):
        super().__init__()

        sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=False)

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                batch_sampler=InfiniteSampler(batch_sampler)))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


class SplitIterator:

    def __init__(self, test_envs):
        self.test_envs = test_envs

    def index_conditional_iterate(self, skip_condition, iterable, index):
        for i, x in enumerate(iterable):
            if skip_condition(i):
                continue
            if index:
                yield i, x
            else:
                yield x

    def train(self, iterable, index=False):
        return self.index_conditional_iterate(
            lambda idx: idx in self.test_envs, iterable, index)

    def test(self, iterable, index=False):
        return self.index_conditional_iterate(
            lambda idx: idx not in self.test_envs, iterable, index)
