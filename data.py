import os
import pickle

import pandas as pd


def get_data(data_files, data_dir):
    train_X = pd.read_csv(os.path.join(os.getcwd(), data_dir, data_files[0]))
    train_y = pd.read_csv(os.path.join(os.getcwd(), data_dir, data_files[1]))
    test_X = pd.read_csv(os.path.join(os.getcwd(), data_dir, data_files[2]))
    return train_X, train_y, test_X


def save_pickle(obj, location):
    with open(location, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(location):
    with open(location, 'rb') as handle:
        return pickle.load(handle)
