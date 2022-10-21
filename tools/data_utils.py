import sys

from keras.datasets import mnist, cifar10
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import keras
import random
import os
import pickle


def get_clean_general(name, network, shape):
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # only used for mnist;
        # y_test = np.expand_dims(y_test, axis=1)
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise Exception(f"No such dataset info:{name}")
    # print(np.min(x_test),np.max(x_test))
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if name == 'cifar100' or (name == 'cifar10' and network == 'resnet20'):
        x_train_mean0 = np.mean(x_train[...,0])
        x_train_mean1 = np.mean(x_train[..., 1])
        x_train_mean2 = np.mean(x_train[..., 2])

        # print(x_train_mean0,x_train_mean1,x_train_mean2)
        x_train[..., 0] -= x_train_mean0
        x_train[..., 1] -= x_train_mean1
        x_train[..., 2] -= x_train_mean2

        x_test[..., 0] -= x_train_mean0
        x_test[..., 1] -= x_train_mean1
        x_test[..., 2] -= x_train_mean2

    x_test = x_test.reshape(shape)
    return x_test, np.squeeze(y_test)


def get_train_general(name, network, shape):
    if name == "mnist":
        (x_train, y_train), (_, _) = mnist.load_data()
    elif name == "cifar10":
        (x_train, y_train),  (_, _) = cifar10.load_data()
    else:
        raise Exception(f"No such dataset info:{name}")
    x_train = x_train.astype('float32') / 255.0
    if name == 'cifar100' or (name == 'cifar10' and network == 'resnet20'):
        x_train_mean0 = np.mean(x_train[..., 0])
        x_train_mean1 = np.mean(x_train[..., 1])
        x_train_mean2 = np.mean(x_train[..., 2])

        x_train[..., 0] -= x_train_mean0
        x_train[..., 1] -= x_train_mean1
        x_train[..., 2] -= x_train_mean2

    x_train = x_train.reshape(shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=get_numclasses(name))
    return x_train, y_train


def get_adv_general(target, name, network):
    data_dict = np.load(os.path.join(f"data/adversarial_examples/full/{name}_{network}",
                                     f"{name}_{network}_{target}_full.npz"))
    inputs, labels = data_dict['inputs'], data_dict['labels']
    return inputs, labels

def get_sa_general(target, dataset, network):
    lsa_rs = np.array(pickle.load(open(f"data/sa_intermediate/{dataset}_{network}_nature_lsa.pkl", "rb")))
    dsa_rs = np.array(pickle.load(open(f"data/sa_intermediate/{dataset}_{network}_nature_dsa.pkl", "rb")))
    substitute = np.load("substitute.npy")
    if target in ['cw', 'bim', 'fgsm', 'jsma']:
        adv_lsa_rs = np.array(pickle.load(open(f"data/sa_intermediate/{dataset}_{network}_{target}_lsa.pkl", "rb")))
        adv_dsa_rs = np.array(pickle.load(open(f"data/sa_intermediate/{dataset}_{network}_{target}_dsa.pkl", "rb")))
        lsa_rs[substitute], dsa_rs[substitute] = adv_lsa_rs[substitute], adv_dsa_rs[substitute]
    return lsa_rs, dsa_rs


def get_candidate_general(target, name, network):
    shape = dataset_shape(name=name)
    candidate_x, candidate_y = get_clean_general(name, network, shape)
    substitute = np.load("substitute.npy")
    if target in ['cw', 'bim', 'fgsm', 'jsma']:
        adv_x, adv_y = get_adv_general(target, name, network)
        candidate_x[substitute], candidate_y[substitute] = adv_x[substitute], adv_y[substitute]
    return candidate_x, candidate_y


def get_finetune_info(target, name, network):
    finetune_matrix = np.load(f'finetuned_prediction_discussion/{name}_{network}/prediction_nature_100.npy').T
    trans_matrix = np.load(f'finetuned_prediction_discussion/{name}_{network}/prediction_trans_nature_100.npy')
    substitute = np.load("substitute.npy")

    if target in ['cw', 'bim', 'fgsm', 'jsma']:
        adv_finetune_matrix = np.load(f'finetuned_prediction_discussion/{name}_{network}/prediction_{target}_100.npy').T
        adv_trans_matrix = np.load(f'finetuned_prediction_discussion/{name}_{network}/prediction_trans_{target}_100.npy')
        finetune_matrix[substitute], trans_matrix[substitute] = adv_finetune_matrix[substitute], adv_trans_matrix[substitute]

    killNum = np.sum(finetune_matrix, axis=1)
    finetune_predict_all = np.argmax(trans_matrix, axis=1)

    return killNum, finetune_predict_all

def get_mutate_info(target, name, network):
    mutate_matrix = np.load(f'mutated_prediction_discussion/{name}_{network}/prediction_nature_100.npy').T
    trans_matrix = np.load(f'mutated_prediction_discussion/{name}_{network}/prediction_trans_nature_100.npy')
    substitute = np.load("substitute.npy")

    if target in ['cw', 'bim', 'fgsm', 'jsma']:
        adv_mutate_matrix = np.load(f'mutated_prediction_discussion/{name}_{network}/prediction_{target}_100.npy').T
        adv_trans_matrix = np.load(f'mutated_prediction_discussion/{name}_{network}/prediction_trans_{target}_100.npy')
        mutate_matrix[substitute], trans_matrix[substitute] = adv_mutate_matrix[substitute], adv_trans_matrix[substitute]

    killNum = np.sum(mutate_matrix, axis=1)
    mutate_predict_all = np.argmax(trans_matrix, axis=1)
    return killNum, mutate_predict_all


def dataset_shape(name):
    dataset_name_dict = {
        'mnist': [-1, 28, 28, 1],
        'cifar10': [-1, 32, 32, 3],
    }
    return dataset_name_dict[name]


def get_numclasses(name):
    if name in ['mnist', 'cifar10']:
        return 10
    else:
        print(f"No num classes for {name}")
        return None

# 混洗数据
def shuffle_data(X, Y, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    return X, Y


def shuffle_data3(X, Y, Z, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y, Z = X[shuffle_indices], Y[shuffle_indices], Z[shuffle_indices]
    return X, Y, Z
