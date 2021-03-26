import torch
import csv

# For Ploting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_learning_curve(loss_record, title=''):
    '''
        Plot learning curve of your DNN (train & dev loss)
    '''

    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::(len(loss_record['train']) // len(loss_record['dev']))]
    figure(figsize=(6, 4))

    # plot curve of training data
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')

    # plot curve of testing data
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    
    # Configurate figure
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylable('MES loss')
    plt.title(f'Learning curve of {title}')
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35, preds=None, targets=None):
    '''
        Plot prediction of your DNN
    '''

    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())

        preds   = torch.cat(pred, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Turth v.s. Prediction')
    plt.show() 


def save_pred(preds, filename):
    '''
        Save predictions to specified file
    '''
    print(f'Saving results to {filename}')
    with open(filename, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])