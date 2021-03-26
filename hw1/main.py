# PyTorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.datasets import COVID19Dataset
from utils.utils import *
from solver import Solver

import os
import argparse
import random
import time
import logging
import numpy as np


def fix_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


def create_folder(config):
    config['save_path']   = os.path.join(config['save_path'], config['model'])
    config['output_path'] = os.path.join(config['output_path'], config['model'])
    if not os.path.isdir(config['save_path']):
        os.makedirs(config['save_path'])
    if not os.path.isdir(config['output_path']):
        os.makedirs(config['output_path'])


def main(config):

    seed = 42067 # set a  random seed for reproducibility
    target_only = False

    # Construct Datasets 
    train_set = COVID19Dataset(config['train_csv'], 'train', target_only=target_only)
    dev_set   = COVID19Dataset(config['dev_csv'], 'dev', target_only=target_only)
    te_set    = COVID19Dataset(config['test.csv'], 'test', target_only=target_only)
    
    # Construct Dataloaders 
    train_loader = DataLoader(
                    train_set,
                    config['batch_size'], 
                    shuffle=True,
                    drop_last=False,
                    num_workers=config['nw'],
                    pin_memory=True
    )
    dev_loader = DataLoader(
                    dev_set,
                    config['batch_size'], 
                    shuffle=False,
                    drop_last=False,
                    num_workers=config['nw'],
                    pin_memory=True
    )
    test_loader = DataLoader(
                    te_set,
                    config['batch_size'], 
                    shuffle=False,
                    drop_last=False,
                    num_workers=config['nw'],
                    pin_memory=True
    )

    solver = Solver(config)

    # Train
    model_loss, model_loss_record = solver.train(train_loader, dev_loader)
    plot_learning_curve(model_loss_record, title='deep model')
    solver.restore_model(f'min_mse.pt')                  # Load best model
    plot_pred(dev_loader, solver.net, config['device'])  # Show prediction on the validation set

    # Test
    preds = solver.test(te_set)
    save_pred(preds, 'pred.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--project', type=str, default='ml2021spring_hw1')
    parser.add_argument('--model', type=str, default='epochs300_maug200_latent1600_lr001_50way_1shot_15query_4layer')
    parser.add_argument('--mode', choices=['train', 'dev', 'test'], default='train')
    parser.add_argument('--train-csv', type=str, default='./covid.train.csv', help="Path of training csv file")
    parser.add_argument('--test-csv', type=str, default='./covid.test.csv', help="Path of testing csv file")
    parser.add_argument('--batch-size', type=int, default=270)
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0, 1 or cpu)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD', help='use SGD / Adam optimizer')
    parser.add_argument('--val-step', type=int, default=10)
    parser.add_argument('--save-step', type=int, default=20)
    parser.add_argument('--save-path', type=str, default='./weights')
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--output-path', type=str, default='./output')
    parser.add_argument('--weights', type=str, default='./weights/vae_lambda0002/best_mse.ckpt', help='path of loaded weight')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')

    # Translate opt from object into a dictioanry
    config = vars(parser.parse_args())

    create_folder(config)

    # Set logging file
    if config['log']:
        logging.basicConfig(filename=os.path.join(config['save_path'], 'log.txt'), level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    mixed_precision = False
    try: # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        logging.info('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
        mixed_precision = False
    
    try:
        import wandb
    except:
        logging.info('Wandb recommended for recording training information')
        config['use_wandb'] = False

    if config['use_wandb']:
        wandb.init(project=config['project'], name=config['model'])
        wandb.config = config

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['nw'] = min([os.cpu_count(), config['batch_size'] if config['batch_size'] > 1 else 0, 6]) # number of workers

    # logging
    for key, value in config.items():
        if config['log']:
            print(f'{key}: {value}')
        logging.info(f'{key}: {value}')

    main(config)



