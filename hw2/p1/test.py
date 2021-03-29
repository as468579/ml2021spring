import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence

print(f'Loading data...')
data_root = './timit_11/'
train = np.load(os.path.join(data_root, 'train_11.npy'))
train_label = np.load(os.path.join(data_root, 'train_label_11.npy')).astype(np.float)
test = np.load(os.path.join(data_root, 'test_11.npy'))

new_train = []
new_test  = []
new_train_label = []

# Split Train Dataset
prev   = np.reshape(train[0], (11, 39))
data   = np.expand_dims(prev[5, :], axis=0)
labels = [ train_label[0] ]
for idx, tr in enumerate(train[1:], start=1):
    tr = np.reshape(tr, (11, 39))
    if not np.array_equal(prev[1:, :], tr[:-1, :]):
        new_train.append(torch.tensor(data))
        new_train_label.append(torch.tensor(labels))

        data   = np.expand_dims(tr[5, :], axis=0)
        labels = [ train_label[idx] ]
    else:
        temp = np.expand_dims(tr[5, :], axis=0)
        data = np.concatenate((data, temp), axis=0)

        labels += [ train_label[idx] ]
    prev = tr

# Split Test Dataset
prev   = np.reshape(test[0], (11, 39))
data   = np.expand_dims(prev[5, :], axis=0)
for idx, te in enumerate(test[1:], start=1):
    te = np.reshape(te, (11, 39))
    if not np.array_equal(prev[1:, :], te[:-1, :]):
        new_test.append(torch.tensor(data))
        data   = np.expand_dims(te[5, :], axis=0)
    else:
        temp = np.expand_dims(te[5, :], axis=0)
        data = np.concatenate((data, temp), axis=0)
    prev = te

# packed_train       = pack_padded_sequence(new_train, train_lens, batch_first=True, enforce_sotred=False)
# packed_train_label = pack_padded_sequence(new_train_label, train_label_lens, batch_first=True, enforce_sotred=False)
# packed_test        = pack_padded_sequence(new_test, test_lens, batch_first=True, enforce_sotred=False)

# train_padded       = pad_packed_sequence(packed_train, batch_first=True)
# train_label_padded = pad_packed_sequence(packed_train_label, batch_first=True)
# test_padded        = pad_packed_sequence(packed_test, backends=True)

padded_train = pad_sequence(new_train, batch_first=True)
padded_train_label = pad_sequence(new_train_label, batch_first=True)
padded_test  = pad_sequence(new_test, batch_first=True)

print(f'Size of training data : {padded_train.shape}')
print(f'Size of training label : {padded_train_label.shape}')
print(f'Size of testing data : {padded_test.shape}')


