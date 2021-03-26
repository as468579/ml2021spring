import torch
import torch.nn as nn
from models import Classifier
from utils.averger import Averager

import os
import logging
class Solver(object):


    def __init__(self, config):

        # Configurations
        self.config = config

        # Build the models
        self.build_models()


    def build_models(self):

        # Models
        self.net = Classifier().to(self.config['device'])

        # Optimizers
        self.optimizer = getattr(torch.optim, self.config['optimizer'])(
            self.net.parameters(),
            lr = self.config['lr'],
        )

        # Citerion
        self.criterion = nn.CrossEntropyLoss()

        # Record
        logging.info(self.net)

    def save_model(self, filename):
        save_path = os.path.join(self.config['save_path'], f'{filename}')
        try:
            logging.info(f'Saved best Neural network ckeckpoints into {save_path}')
            torch.save(self.net.state_dict(), save_path, _use_new_zipfile_serialization=False)
        except:
            logging.error(f'Error saving weights to {save_path}')

    def restore_model(self, filename):
        weight_path = os.path.join(self.config['save_path'], f'{filename}')
        try:
            logging.info(f'Loading the trained Extractor from {weight_path}')
            self.net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc:storage))

        except:
            logging.error(f'Error loading weights from {weight_path}')


    def train(self, tr_set, val_set):
        
        num_train_set = len(tr_set.dataset)
        num_val_set   = len(val_set.dataset)

        best_acc = 0.0
        for epoch in range(1, self.config['n_epochs']+1):
            record = {}
            train_acc  = 0.0
            train_loss = 0.0
            val_acc    = 0.0
            val_loss   = 0.0

            # Start training
            self.net.train()
            for i_batch, batch in enumerate(tr_set, start=1):

                # if i_batch != len(tr_set):
                #     print(f"[{epoch}/{self.config['n_epochs']}] {i_batch}/{len(tr_set)} iters", end='\r')
                # else:
                #     print(f"[{epoch}/{self.config['n_epochs']}] {i_batch}/{len(tr_set)} iters")

                print(f"[{epoch:3d}/{self.config['n_epochs']}] {i_batch}/{len(tr_set)} iters", end='\r')

                x, y = batch[0].to(self.config['device']), batch[1].to(self.config['device'])
                pred = self.net(x)
                batch_loss = self.criterion(pred, y)
                _, train_pred = torch.max(pred, dim=1)

                # Update model
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Record
                train_acc += (train_pred.cpu() == y.cpu()).sum().item()
                train_loss += batch_loss.item()

            train_acc /=  num_train_set
            train_loss /= len(tr_set)

            record['train_acc'] = train_acc
            record['train_loss'] = train_loss

            # After each epoch, test your model on the validation set
            self.net.eval()
            with torch.no_grad():
                for x, y in val_set:

                    x, y = x.to(self.config['device']), y.to(self.config['device'])
                    pred = self.net(x)
                    batch_loss = self.criterion(pred, y)
                    _, val_pred = torch.max(pred, dim=1)

                    val_acc += (val_pred.cpu() == y.cpu()).sum().item()
                    val_loss += batch_loss.item()

                val_acc  /= num_val_set
                val_loss /= len(val_set)

                record['val_acc'] = val_acc
                record['val_loss'] = val_loss
                logging.info(f'[{epoch:3d}/{self.config["n_epochs"]}] Train Acc: {train_acc:3.6f} Loss: {train_loss:3.6f} | Val Acc: {val_acc:3.6f} Loss: {val_loss:3.6f}')


            if self.config['use_wandb']:
                import wandb
                wandb.log(record)

            # Save a chekcpoint if model improves
            if val_acc > best_acc:
                # Save model if model improved
                best_acc = val_acc
                print(f'Saving model (epoch = {epoch:3d}, val acc = {val_acc:.3f})')
                self.save_model(f'best_acc_{int(val_acc * 100)}.pt')

            if (epoch % self.config['save_step'] == 0) and (epoch != self.config['n_epochs']):
                print(f'Saving model (epoch = {epoch:3d}, val acc = {val_acc:.3f})')
                self.save_model(f'epoch_{epoch}.pt')

                           
                
        print(f'Saving model (epoch = {epoch:3d}, acc = {val_acc:.3f}')
        self.save_model(f'last.pt')

        print(f'Finished training after {epoch} epochs.')
        return


    def test(self, te_set):
        self.net.eval()
        preds = []
        with torch.no_grad():
            for data in te_set:
                x = data.to(self.config['device'])
                pred = self.net(x)
                _, test_pred = torch.max(pred, dim=1)

                for y in test_pred.cpu().numpy():
                    preds += [y]

        with open(self.config['output_csv'], w) as f:
            f.write('Id,Class\n')
            for i, y in enumerate(preds):
                f.write(f'{i}, {y}\n')
        
