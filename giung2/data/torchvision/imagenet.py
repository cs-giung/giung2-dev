import os
import numpy as np

import torch
from timm.data import ImageDataset
from timm.data.transforms_factory import (
    transforms_noaug_train, transforms_imagenet_train, transforms_imagenet_eval)


_TRANSFORMS_FACTORY = {
    'eval': transforms_imagenet_eval(
        interpolation='bicubic', use_prefetcher=True),
    'preset0': transforms_noaug_train(
        interpolation='bicubic', use_prefetcher=True),
    'preset1': transforms_imagenet_train(
        interpolation='bicubic', use_prefetcher=True),
    }


def project_logits(logits, classmasks):
    if isinstance(logits, list):
        return [project_logits(l, classmasks) for l in logits]
    if logits.shape[1] > sum(classmasks):
        return logits[:, classmasks]
    else:
        return logits


class ImageNet:
    def __init__(self, root='~/data/', preset='preset1', load_trn=True,
                 batch_size=256, num_workers=32, prefetch_factor=2):
        self.root            = root
        self.preset          = preset
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        
        self.trn_transform   = _TRANSFORMS_FACTORY[preset]
        self.val_transform   = _TRANSFORMS_FACTORY['eval']

        # setup datasets
        if load_trn:
            self.trn_dataset = ImageDataset(
                self.get_trn_directory(), transform=self.trn_transform)
        self.val_dataset = ImageDataset(
            self.get_val_directory(), transform=self.val_transform)

    def trn_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trn_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, prefetch_factor=self.prefetch_factor,
            drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, prefetch_factor=self.prefetch_factor,
            drop_last=False)

    def get_trn_directory(self):
        return os.path.join(self.root, 'imagenet/train')

    def get_val_directory(self):
        return os.path.join(self.root, 'imagenet/val')


class ImageNetV2(ImageNet):
    def get_val_directory(self):
        return os.path.join(self.root, 'imagenet-v2')


class ImageNetSketch(ImageNet):
    def get_val_directory(self):
        return os.path.join(self.root, 'imagenet-sketch')


class ImageNetA(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_sublist = [
              6,  11,  13,  15,  17,  22,  23,  27,  30,  37,  39,  42,  47,
             50,  57,  70,  71,  76,  79,  89,  90,  94,  96,  97,  99, 105,
            107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207,
            234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308,
            309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330,
            334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402,
            404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456,
            457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528,
            530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575,
            579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658,
            668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
            763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803,
            804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859,
            862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933,
            934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980,
            981, 984, 986, 987, 988]
        self.classmasks = [(i in self.class_sublist) for i in range(1000)]

    def get_val_directory(self):
        return os.path.join(self.root, 'imagenet-a')

    def project_logits(self, logits):
        return project_logits(logits, self.classmasks)


class ImageNetR(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_sublist = [
              1,   2,   4,   6,   8,   9,  11,  13,  22,  23,  26,  29,  31,
             39,  47,  63,  71,  76,  79,  84,  90,  94,  96,  97,  99, 100,
            105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151,
            155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207,
            208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259,
            260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293,
            296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330,
            334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365,
            366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425,
            428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471,
            472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593,
            594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724,
            763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833,
            847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934,
            936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
            967, 980, 981, 983, 988]
        self.classmasks = [(i in self.class_sublist) for i in range(1000)]

    def get_val_directory(self):
        return os.path.join(self.root, 'imagenet-r')

    def project_logits(self, logits):
        return project_logits(logits, self.classmasks)
