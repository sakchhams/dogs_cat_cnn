from __future__ import print_function
from load_data import DataLoader
from tf_model import start_train

train_data_loader = DataLoader()
start_train(train_data_loader)