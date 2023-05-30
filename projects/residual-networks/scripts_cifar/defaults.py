import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def default_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root', default='./data/', type=str,
        help='path to the directory (default: ./data/)')
    parser.add_argument(
        '--data_name', default='CIFAR10_x32', type=str,
        choices=[
            'CIFAR10_x32', 'CIFAR100_x32',
            'TinyImageNet200_x32', 'TinyImageNet200_x64'])

    parser.add_argument(
        '--resnet_depth', default=20, type=int,
        help='depth of the residual network (default: 20)')
    parser.add_argument(
        '--resnet_width', default=1, type=int,
        help='width of the residual network (default: 1)')

    parser.add_argument(
        '--batch_size', default=256, type=int,
        help='the number of examples for each mini-batch (default: 256)')
    parser.add_argument(
        '--num_workers', default=32, type=int,
        help='how many subprocesses to use for data loading (default: 32)')
    parser.add_argument(
        '--prefetch_factor', default=2, type=int,
        help='number of batches loaded in advance by each worker (default: 2)')

    return parser
