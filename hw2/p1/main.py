# PyTorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.datasets import TIMITDataset
from solver import Solver

import os
import argparse
import random
import time
import logging
import numpy as np
import gc


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

def prepare_data():

    print(f'Loading data...')
    data_root = './timit_11/'
    train = np.load(os.path.join(data_root, 'train_11.npy'))
    train_label = np.load(os.path.join(data_root, 'train_label_11.npy'))
    test = np.load(os.path.join(data_root, 'test_11.npy'))

    print(f'Size of training data : {train.shape}')
    print(f'Size of testing data : {test.shape}')
    return train, train_label, test


def main(config):

    fix_seeds(0) # set a  random seed for reproducibility
    
    train, train_label, test = prepare_data()

    # Split data into training, validation set
    VAL_RATIO = 0.2
    percent = int(train.shape[0] * (1 - VAL_RATIO))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    print(f'Size of training set : {train_x.shape}')
    print(f'Size of validation set : {val_x.shape}')

    # Construct Datasets 
    train_set = TIMITDataset(train_x, train_y)
    val_set   = TIMITDataset(val_x, val_y)
    test_set  = TIMITDataset(test, None)
    
    # Construct Dataloaders 
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # Clean up unneeded variables to save memory
    del train, train_label, train_x, train_y, val_x, val_y
    gc.collect()


    solver = Solver(config)

    # Train
    solver.train(train_loader, val_loader)

    # Test
    # solver.restore_model(f'last.pt')                  # Load best model
    # preds = solver.test(test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--project', type=str, default='ml2021spring_hw2_p1')
    parser.add_argument('--model', type=str, default='dropout')
    parser.add_argument('--mode', choices=['train', 'dev', 'test'], default='train')
    parser.add_argument('--train-csv', type=str, default='./covid.train.csv', help="Path of training csv file")
    parser.add_argument('--test-csv', type=str, default='./covid.test.csv', help="Path of testing csv file")
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0, 1 or cpu)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='use SGD / Adam optimizer')
    parser.add_argument('--val-step', type=int, default=10)
    parser.add_argument('--save-step', type=int, default=20)
    parser.add_argument('--save-path', type=str, default='./weights')
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--output_csv', type=str, default='preidction.csv', help="Output filename")
    parser.add_argument('--output-path', type=str, default='./output')
    parser.add_argument('--weights', type=str, default='', help='path of loaded weight')
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



