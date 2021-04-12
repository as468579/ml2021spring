import torch
import torch.nn as nn
import numpy as np
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
        self.criterion = nn.CrossEntropyLoss(reduce=False)

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

def get_pseudo_labels(self, unlabel_set, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # Make sure the model is in eval mode.
    self.net.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    for i_batch, batch in enumerate(unlabel_set, start=1):
        img = batch[0].to(self.config['device'])

        # Forward the data, using torch.no_grad() accelerates the forward process.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = self.model(img)

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.

    # # Turn off the eval mode.
    model.train()
    return dataset

    def train(self, tr_set, val_set):
        
        num_train_set = len(tr_set.dataset)
        num_val_set   = len(val_set.dataset)

        best_acc = 0.0
        for epoch in range(1, self.config['n_epochs']+1):
            record = {}
            train_acc  = Averager()
            train_loss = Averager()
            val_acc    = Averager()
            val_loss   = Averager()

            # Start training
            self.net.train()
            for i_batch, batch in enumerate(tr_set, start=1):

                print(f"[{epoch:3d}/{self.config['n_epochs']}] {i_batch}/{len(tr_set)} iters", end='\r')

                x, y = batch[0].to(self.config['device']), batch[1].to(self.config['device'])
                pred = self.net(x)

                # Calculate loss
                is_target = torch.sign(torch.abs(torch.sum(x, dim=-1)))
                batch_loss = self.criterion(pred.view(-1, 39), y.view(-1))
                batch_loss = batch_loss.view(is_target.shape[0], -1)
                batch_loss *= is_target
                batch_loss = torch.sum(batch_loss)

                _, train_pred = torch.max(pred, dim=-1)
                # Update model
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Record
                num_target = is_target.cpu().sum().item()
                acc = ((train_pred.cpu() == y.cpu()) * is_target.cpu()).sum().item()
                train_acc.add(acc, num_target)
                train_loss.add(batch_loss.item(), num_target)

            train_acc = train_acc.getValue()
            train_loss = train_loss.getValue()

            record['train_acc'] = train_acc
            record['train_loss'] = train_loss

            # After each epoch, test your model on the validation set
            self.net.eval()
            with torch.no_grad():
                for x, y in val_set:

                    x, y = x.to(self.config['device']), y.to(self.config['device'])
                    pred = self.net(x)

                    # Calculate loss
                    is_target = torch.sign(torch.abs(torch.sum(x, dim=-1)))
                    batch_loss = self.criterion(pred.view(-1, 39), y.view(-1))
                    batch_loss = batch_loss.view(is_target.shape[0], -1)
                    batch_loss *= is_target
                    batch_loss = torch.sum(batch_loss)

                    _, val_pred = torch.max(pred, dim=-1)

                    num_target = is_target.cpu().sum().item()
                    acc = ((val_pred.cpu() == y.cpu()) * is_target.cpu()).sum().item()
                    val_acc.add(acc, num_target)
                    val_loss.add(batch_loss.item(), num_target)

                val_acc  = val_acc.getValue()
                val_loss = val_loss.getValue()

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

    def gussian(self, pred_prob):
        gussian_kernel = np.array([0.000000003244555, 0.007462608761363, 0.985074775988165, 0.007462608761363, 0.000000003244555])
        pred_prob = pred_prob.cpu().numpy()
        for i_batch in range(pred_prob.shape[0]):
            for idx in range(39):
                pred_prob[i_batch, 2:-2, idx] = np.convolve(pred_prob[i_batch, 2:-2, idx], gussian_kernel, 'same')
        
        return torch.tensor(pred_prob)

    def test(self, te_set, threshold=0.6):
        self.net.eval()
        preds = []
        pseudo_train = []
        pseudo_label = []
        with torch.no_grad():
            for x in te_set:
                x = x.to(self.config['device'])
                pred = self.net(x)
                # pred = self.gussian(pred)
                pred_prob, test_pred = torch.max(pred, dim=-1)


                is_target = torch.sign(torch.abs(torch.sum(x, dim=-1)))
                
                # Store pseudo training data 
                for i_batch, batch in enumerate(x.cpu().numpy()):
                    new_train = torch.tensor([ p for idx, p in enumerate(batch) if (is_target[i_batch, idx]) and pred_prob[i_batch, idx] > threshold])
                    pseudo_train.append(new_train)

                # Store result and pseudo training label
                for i_batch, batch in enumerate(test_pred.cpu().numpy()):
                    preds += [p for idx, p in enumerate(batch) if is_target[i_batch, idx] ]
                    new_label = torch.tensor([ l for idx, l in enumerate(batch) if(is_target[i_batch, idx] and pred_prob[i_batch, idx] > threshold) ])
                    pseudo_label.append(new_label)

        with open(self.config['output_csv'], 'w') as f:
            f.write('Id,Class\n')
            for i, y in enumerate(preds):
                f.write(f'{i},{y}\n')

        return pseudo_train, pseudo_label
        
