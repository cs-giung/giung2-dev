import os
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

    parser.add_argument('--data_root', default='~/data/', type=lambda e: os.path.expanduser(e),
                        help='root directory containing datasets (default: ~/data/)')
    
    parser.add_argument('--clip_name', default='openai/clip-vit-base-patch16', type=str,
                        help='see https://huggingface.co/models?filter=clip (default: openai/clip-vit-base-patch16)')
    parser.add_argument('--clip_zero_head', default=False, type=str2bool,
                        help='initialize head weights to zeros (default: False)')

    parser.add_argument('--batch_size', default=256, type=int,
                        help='the number of examples for each mini-batch (default: 256)')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='the number of workers for dataloaders (default: 32)')
    
    return parser
