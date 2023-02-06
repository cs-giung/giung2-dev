import os
import torch
import torchvision
from .classnames import openai_classnames


def project_logits(logits, class_sublist_mask):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask) for l in logits]
    if logits.shape[1] > sum(class_sublist_mask):
        return logits[:, class_sublist_mask]
    else:
        return logits


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {'images': image, 'labels': label, 'image_paths': self.samples[index][0]}


class ImageNet:

    def __init__(self, preprocess, location=os.path.expanduser('~/data/'), batch_size=32, num_workers=32):
        self.preprocess  = preprocess
        self.location    = location
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.classnames  = openai_classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        self.train_dataset = ImageFolderWithPaths(
            path      = os.path.join(self.location, 'imagenet/train'),
            transform = self.preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, sampler=None, batch_size=self.batch_size,
            num_workers=self.num_workers, drop_last=True, shuffle=True)

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, sampler=None, batch_size=self.batch_size,
            num_workers=self.num_workers, drop_last=False, shuffle=False)

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def get_test_path(self):
        return os.path.join(self.location, 'imagenet/val')


class ImageNetSubsample(ImageNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits):
        return project_logits(logits, self.class_sublist_mask)
