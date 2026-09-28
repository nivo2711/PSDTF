import os
import sys
from argparse import ArgumentParser


import torch

import basicts

torch.set_num_threads(4)

def parse_args():
    parser = ArgumentParser(description='Train the PDSTF traffic flow forecasting model with BasicTS.')
    parser.add_argument('-c', '--cfg', default='baselines/PDSTF/PEMS04.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()
