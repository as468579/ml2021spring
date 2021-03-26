import torch
from models import NeuralNet

import os
import logging
class Solver(object):


    def __init__(self, config):

        # Configurations
        self.config = config

        # Build the models
        self.build_models()

    def build_models(self):
        self.net = NeuralNet().to(self.config['device'])
        self.optimizer = getattr(torch.optim, self.config['optimizer'])(
            self.net.parameters(),
            lr = self.config['lr'],
            momentum = self.config['momentum']
        )

    def save_model(self, filename):
        save_path = os.path.join(self.save_path, f'{filename}')
        try:
            logging.info(f'Saved best Neural network ckeckpoints into {save_path}')
            torch.save(self.net.state_dict(), save_path, _use_new_zipfile_serialization=False)
        except:
            logging.error(f'Error saving weights to {save_path}')

    def restore_models(self, filename):
        weight_path = os.path.join(self.save_path, f'{filename}')
        try:
            logging.info(f'Loading the trained Extractor from {weight_path}')
            self.extractor.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc:storage))

        except:
            logging.error(f'Error loading weights from {weight_path}')


    def train(self, tr_set, dv_set):

        min_mse = 1000.
        loss_record = {'train':[], 'dev': []}
        early_stop_cnt = 0
        
        epoch = 1
        while epoch <= self.config['n_epochs']:

            # Start training
            self.net.train()
            for i_batch, batch in enumerate(tr_set, start=1):

                if i_batch != len(tr_set):
                    print(f" {i_batch+1}/{len(tr_set)} iters", end='\r')
                else:
                    print(f" {i_batch+1}/{len(tr_set)} iters")

                x, y = batch[0].to(self.config['device']), batch[1].to(self.config['device'])
                pred = self.net(x)
                mse_loss = self.net(pred, y)

                # Update model
                self.optimzier.zero_grad()
                mse_loss.backward()
                self.optimzier.step()

                # Record
                loss_record['train'] += [mse_loss.detach().cpu().item()]

            # After each epoch, test your model on the validation set
            self.net.eval()
            dev_loss = 0.0
            for x, y in dv_set:

                x, y = x.to(self.config['device']), y.to(self.config['device'])
                with torch.no_grad():
                    pred = self.net(x)
                    mse_loss = self.net.cal_loss(pred, y)
                dev_loss = mse_loss.detach().cpu().item() * len(x)

            dev_loss = dev_loss / len(dv_set.dataset)


            if dev_loss < min_mse:
                # Save model if model improved
                min_mse = dev_loss
                print(f'Saving model (epoch = {epoch:4d}, loss = {min_mse:.4f})')
                self.save_model(f'min_mse.pt')
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            epoch += 1
            loss_record['dev'] += [dev_loss]
            if early_stop_cnt > self.config['early_stop']:
                # Stop training if your model stops imporving for "config['early_stop']" epochs. 
                break

        print(f'Finished training after {epoch} epochs.')
        return min_mse, loss_record


    def test(self, te_set):
        self.net.eval()
        preds = []
        for data in te_set:
            x = data.to(self.config['device'])
            with torch.no_grad():
                pred = self.net(x)
                preds += [pred.detach().cpu()]

        preds = torch.cat(preds, dim=0).numpy()
        return preds
        
