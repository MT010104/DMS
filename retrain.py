<<<<<<< HEAD
import os
import argparse
import sys
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from ATS.ATS import ATS
import metrics

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, adadelta

from tools.data_utils import dataset_shape, get_sa_general
from tools.data_utils import get_candidate_general, get_train_general
from tools.data_utils import get_finetune_info, get_mutate_info
from tools.model_utils import load_model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def load_model_again(dataset, network):
    model = load_model(network=network, dataset=dataset)
    # model = frozen_layers(model) # 冻结部分层
    if network == 'vgg16':
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['accuracy'])
    return model


def frozen_layers(model):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            break
        layer.trainable = False
    return model


def train_model(model, filepath, x_sel, y_sel, epochs=10, verbose=0):
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy', mode='auto',
                                 save_weights_only=False)
    model.fit(x_sel, y_sel, batch_size=128, epochs=epochs, validation_data=(x_val, y_val),
              callbacks=[checkpoint],
              verbose=verbose)
    return model


def retrain(model, x, y, filepath):
    x = np.vstack((x, x_train[:len(x)]))
    print(y.shape)
    print(y_train.shape)
    y = np.vstack((y, y_train[:len(y)]))
    trained_model = train_model(model, filepath, x, y, 10)
    acc_val_new = trained_model.evaluate(x_val, y_val)[1]
    print("retrain model path: {}".format(filepath))
    print("train acc improve {} -> {}".format(acc_val0, acc_val_new))
    return acc_val_new


def exp_ats():
    ats = ATS()
    nb_classes = 10
    rank_lst, _, _ = ats.get_priority_sequence(x_test, y_test_psedu, nb_classes, model, th=0.001)
    return rank_lst


def exp_coverage(std=0):

    input = model.input

    # Dense & Conv
    layer_type, layer_output = [], []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_type.append("conv")
            layer_output.append(layer.output)
        if isinstance(layer, keras.layers.Dense):
            layer_type.append("dense")
            layer_output.append(layer.output)
    layers = list(zip(layer_type, layer_output))

    ac = metrics.nac(x_test, input, layers, t=0.5)
    rank_ac = ac.rank_2(x_test)

    bc = metrics.nbc(x_train, input, layers, std=std)
    rank_bc = bc.rank_2(x_test, use_lower=True)

    rank_snac = bc.rank_2(x_test, use_lower=False)

    tk = metrics.tknc(x_test, input, layers, k=3)
    rank_tk = tk.rank(x_test)

    return rank_ac, ac.rank_fast(x_test), rank_bc, rank_snac, rank_tk


def exp_sa(target):
    lsa, dsa = get_sa_general(target, dataset, network)
    rank_lsa = np.argsort(-lsa[shuffle_indices][:all_sample_size])
    rank_dsa = np.argsort(-dsa[shuffle_indices][:all_sample_size])
    return rank_lsa, rank_dsa


def exp_mutate(target):
    killNum, mutate_predict_all = get_mutate_info(target, dataset, network)
    killNum = killNum[shuffle_indices][:all_sample_size]
    return np.argsort(-killNum)


def exp_gini():
    pred_test_prob = model.predict(x_test)
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    return rank_lst


def exp_my(target, class_Dic, selectedSize):

    killNum, finetune_predict_all = get_finetune_info(target, dataset, network)
    killNum = killNum[shuffle_indices][:all_sample_size]
    selectedIdx = []
    spare = []
    for label in class_Dic:

        candidates = np.array(class_Dic[label])
        selectedSizeFromClass = int(np.around(len(candidates) / all_sample_size * selectedSize))

        subclass_Dic = {}
        for outer in candidates:
            majorLabel1 = finetune_predict_all[outer]
            if majorLabel1 == label:
                spare.append(outer)
            else:
                if majorLabel1 not in subclass_Dic:
                    subclass_Dic[majorLabel1] = []
                subclass_Dic[majorLabel1].append(outer)

        for label2 in subclass_Dic:
            subclass_Dic[label2] = np.array(subclass_Dic[label2])
            subclass_Dic[label2] = subclass_Dic[label2][np.argsort(-killNum[subclass_Dic[label2]])]
            subSelectedSize = max(
                int(np.around(len(subclass_Dic[label2]) / len(candidates) * selectedSizeFromClass)), 1)
            selectedIdx.extend(subclass_Dic[label2][:subSelectedSize])
            if subSelectedSize < len(subclass_Dic[label2]):
                spare.extend(subclass_Dic[label2][subSelectedSize:])

    spare = np.array(spare)
    spare = spare[np.argsort(-killNum[spare])]
    if len(selectedIdx) < selectedSize:
        extraSize = selectedSize - len(selectedIdx)
        selectedIdx.extend(spare[:extraSize])
    return selectedIdx


def exp_my2(target):
    killNum, mutate_predict_all = get_finetune_info(target, dataset, network)
    rank_lst = np.argsort(-killNum[shuffle_indices][:all_sample_size])
    return rank_lst


def exp_deepest(target, num):

    origin_data = np.array(pd.read_csv(f'data/deepest_info/{dataset}_{network}_{target}.csv'))
    chosen_data = origin_data[shuffle_indices][:all_sample_size]
    df = pd.DataFrame(np.array(chosen_data))
    df.to_csv(os.path.join('data/deepest_info', f"{dataset}_{network}_{target}_retrain.csv"), index=False)

    res_combo = os.popen(f"java -jar deepest.jar data/deepest_info/{dataset}_{network}_{target}_retrain.csv combo 0.70 {num}")
    text_combo = res_combo.read()

    line = text_combo.split("\n")[2]
    info = line.split(':')[-1].strip()[1:-1]
    selectedIdx = []
    for i in info.split(','):
        selectedIdx.append(int(i))

    res_combo.close()
    return selectedIdx


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset",
                        type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network",
                        type=str, default="lenet1")
    parser.add_argument("--ratio", "-r", type=float, default=0.5)
    args = parser.parse_args()

    dataset = args.dataset
    network = args.network

    global model, x_test, y_test, y_test_psedu, shuffle_indices, all_sample_size, \
        x_val, y_val, x_train, y_train
    save_path = f"results/retrain"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape = dataset_shape(name=dataset)
    x_train, y_train = get_train_general(name=dataset, network=network, shape=shape)
    shuffle_train_indices = np.random.permutation(np.arange(len(x_train)))
    x_train, y_train = x_train[shuffle_train_indices], y_train[shuffle_train_indices]


    data, index = [], []
    columns = [1000]
    baselines = ['nac', 'nac2', 'nbc', 'snac', 'tknc', 'lsa', 'dsa', 'deepgini', 'ats', 'deepest', 'random', 'mutate', 'my*', 'my']

    for target in ['cw', 'bim', 'fgsm', 'jsma']:
    # for target in ['cw']:

        index.extend([target+'_'+m for m in baselines])
        model = load_model_again(dataset, network)
        x_dau, y_dau = get_candidate_general(target, dataset, network)
        shuffle_indices = np.random.permutation(np.arange(len(x_dau)))
        x_dau, y_dau = x_dau[shuffle_indices], y_dau[shuffle_indices]
        dau_size = x_dau.shape[0]

        all_sample_size = int(dau_size*args.ratio)
        x_test, y_test = x_dau[:all_sample_size], y_dau[:all_sample_size]
        x_val, y_val = x_dau[all_sample_size:], y_dau[all_sample_size:]

        y_test_psedu = get_psedu_label(model, x_test)
        acc_val0 = np.sum(y_test == y_test_psedu)/all_sample_size
        y_test = keras.utils.to_categorical(y_test, 10)
        y_val = keras.utils.to_categorical(y_val, 10)

        rank_ats = exp_ats()
        rank_nac, rank_nac2, rank_nbc, rank_snac, rank_tknc = exp_coverage()
        rank_lsa, rank_dsa = exp_sa(target)
        rank_mutate = exp_mutate(target)
        rank_gini = exp_gini()
        rank_my2 = exp_my2(target)
        rank_random = np.arange(all_sample_size)
        np.random.shuffle(rank_random)

        class_Dic = {}
        for i, label in enumerate(y_test_psedu):
            if label not in class_Dic:
                class_Dic[label] = []
            class_Dic[label].append(i)

        acc_nac, acc_nbc, acc_snac, acc_tknc = [], [], [], []
        acc_lsa, acc_dsa, acc_ats, acc_random = [], [], [], []
        acc_deepest, acc_mutate, acc_my2, acc_my = [], [], [], []
        acc_gini, acc_nac2 = [], []

        for num in columns:

            filepath = f"retrained_model/{dataset}_{network}_nac_{num}.h5"
            acc_val = retrain(model, x_test[rank_nac][:num], y_test[rank_nac][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nac.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_nac2_{num}.h5"
            acc_val = retrain(model, x_test[rank_nac2][:num], y_test[rank_nac2][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nac2.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_nbc_{num}.h5"
            acc_val = retrain(model, x_test[rank_nbc][:num], y_test[rank_nbc][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nbc.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_snac_{num}.h5"
            acc_val = retrain(model, x_test[rank_snac][:num], y_test[rank_snac][:num], filepath)
            model = load_model_again(dataset, network)
            acc_snac.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_tknc_{num}.h5"
            acc_val = retrain(model, x_test[rank_tknc][:num], y_test[rank_tknc][:num], filepath)
            model = load_model_again(dataset, network)
            acc_tknc.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_lsa_{num}.h5"
            acc_val = retrain(model, x_test[rank_lsa][:num], y_test[rank_lsa][:num], filepath)
            model = load_model_again(dataset, network)
            acc_lsa.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_dsa_{num}.h5"
            acc_val = retrain(model, x_test[rank_dsa][:num], y_test[rank_dsa][:num], filepath)
            model = load_model_again(dataset, network)
            acc_dsa.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_gini_{num}.h5"
            acc_val = retrain(model, x_test[rank_gini][:num], y_test[rank_gini][:num], filepath)
            model = load_model_again(dataset, network)
            acc_gini.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_ats_{num}.h5"
            acc_val = retrain(model, x_test[rank_ats][:num], y_test[rank_ats][:num], filepath)
            model = load_model_again(dataset, network)
            acc_ats.append(round(acc_val - acc_val0, 4) * 100)

            selectedIdx_deepest = exp_deepest(target, num)
            filepath = f"retrained_model/{dataset}_{network}_deepest_{num}.h5"
            acc_val = retrain(model, x_test[selectedIdx_deepest], y_test[selectedIdx_deepest], filepath)
            model = load_model_again(dataset, network)
            acc_deepest.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_random_{num}.h5"
            acc_val = retrain(model, x_test[rank_random][:num], y_test[rank_random][:num], filepath)
            model = load_model_again(dataset, network)
            acc_random.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_mutate_{num}.h5"
            acc_val = retrain(model, x_test[rank_mutate[:num]], y_test[rank_mutate[:num]], filepath)
            model = load_model_again(dataset, network)
            acc_mutate.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_my2_{num}.h5"
            acc_val = retrain(model, x_test[rank_my2][:num], y_test[rank_my2][:num], filepath)
            model = load_model_again(dataset, network)
            acc_my2.append(round(acc_val - acc_val0, 4) * 100)

            selectedIdx_my = exp_my(target, class_Dic, num)
            filepath = f"retrained_model/{dataset}_{network}_my_{num}.h5"
            acc_val = retrain(model, x_test[selectedIdx_my], y_test[selectedIdx_my], filepath)
            model = load_model_again(dataset, network)
            acc_my.append(round(acc_val - acc_val0, 4) * 100)

        data.append(acc_nac)
        data.append(acc_nac2)
        data.append(acc_nbc)
        data.append(acc_snac)
        data.append(acc_tknc)
        data.append(acc_lsa)
        data.append(acc_dsa)
        data.append(acc_gini)
        data.append(acc_ats)
        data.append(acc_deepest)
        data.append(acc_random)
        data.append(acc_mutate)
        data.append(acc_my2)
        data.append(acc_my)

    df = pd.DataFrame(np.array(data), columns=columns, index=index)
    df.to_csv(os.path.join(save_path, f"{dataset}_{network}_retrain_{args.ratio}.csv"))
=======
import os
import argparse
import sys
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from ATS.ATS import ATS
import metrics

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, adadelta

from tools.data_utils import dataset_shape, get_sa_general
from tools.data_utils import get_candidate_general, get_train_general
from tools.data_utils import get_finetune_info, get_mutate_info
from tools.model_utils import load_model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def load_model_again(dataset, network):
    model = load_model(network=network, dataset=dataset)
    # model = frozen_layers(model) # 冻结部分层
    if network == 'vgg16':
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['accuracy'])
    return model


def frozen_layers(model):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            break
        layer.trainable = False
    return model


def train_model(model, filepath, x_sel, y_sel, epochs=10, verbose=0):
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy', mode='auto',
                                 save_weights_only=False)
    model.fit(x_sel, y_sel, batch_size=128, epochs=epochs, validation_data=(x_val, y_val),
              callbacks=[checkpoint],
              verbose=verbose)
    return model


def retrain(model, x, y, filepath):
    x = np.vstack((x, x_train[:len(x)]))
    print(y.shape)
    print(y_train.shape)
    y = np.vstack((y, y_train[:len(y)]))
    trained_model = train_model(model, filepath, x, y, 10)
    acc_val_new = trained_model.evaluate(x_val, y_val)[1]
    print("retrain model path: {}".format(filepath))
    print("train acc improve {} -> {}".format(acc_val0, acc_val_new))
    return acc_val_new


def exp_ats():
    ats = ATS()
    nb_classes = 10
    rank_lst, _, _ = ats.get_priority_sequence(x_test, y_test_psedu, nb_classes, model, th=0.001)
    return rank_lst


def exp_coverage(std=0):

    input = model.input

    # Dense & Conv
    layer_type, layer_output = [], []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_type.append("conv")
            layer_output.append(layer.output)
        if isinstance(layer, keras.layers.Dense):
            layer_type.append("dense")
            layer_output.append(layer.output)
    layers = list(zip(layer_type, layer_output))

    ac = metrics.nac(x_test, input, layers, t=0.5)
    rank_ac = ac.rank_2(x_test)

    bc = metrics.nbc(x_train, input, layers, std=std)
    rank_bc = bc.rank_2(x_test, use_lower=True)

    rank_snac = bc.rank_2(x_test, use_lower=False)

    tk = metrics.tknc(x_test, input, layers, k=3)
    rank_tk = tk.rank(x_test)

    return rank_ac, ac.rank_fast(x_test), rank_bc, rank_snac, rank_tk


def exp_sa(target):
    lsa, dsa = get_sa_general(target, dataset, network)
    rank_lsa = np.argsort(-lsa[shuffle_indices][:all_sample_size])
    rank_dsa = np.argsort(-dsa[shuffle_indices][:all_sample_size])
    return rank_lsa, rank_dsa


def exp_mutate(target):
    killNum, mutate_predict_all = get_mutate_info(target, dataset, network)
    killNum = killNum[shuffle_indices][:all_sample_size]
    return np.argsort(-killNum)


def exp_gini():
    pred_test_prob = model.predict(x_test)
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    return rank_lst


def exp_my(target, class_Dic, selectedSize):

    killNum, finetune_predict_all = get_finetune_info(target, dataset, network)
    killNum = killNum[shuffle_indices][:all_sample_size]
    selectedIdx = []
    spare = []
    for label in class_Dic:

        candidates = np.array(class_Dic[label])
        selectedSizeFromClass = int(np.around(len(candidates) / all_sample_size * selectedSize))

        subclass_Dic = {}
        for outer in candidates:
            majorLabel1 = finetune_predict_all[outer]
            if majorLabel1 == label:
                spare.append(outer)
            else:
                if majorLabel1 not in subclass_Dic:
                    subclass_Dic[majorLabel1] = []
                subclass_Dic[majorLabel1].append(outer)

        for label2 in subclass_Dic:
            subclass_Dic[label2] = np.array(subclass_Dic[label2])
            subclass_Dic[label2] = subclass_Dic[label2][np.argsort(-killNum[subclass_Dic[label2]])]
            subSelectedSize = max(
                int(np.around(len(subclass_Dic[label2]) / len(candidates) * selectedSizeFromClass)), 1)
            selectedIdx.extend(subclass_Dic[label2][:subSelectedSize])
            if subSelectedSize < len(subclass_Dic[label2]):
                spare.extend(subclass_Dic[label2][subSelectedSize:])

    spare = np.array(spare)
    spare = spare[np.argsort(-killNum[spare])]
    if len(selectedIdx) < selectedSize:
        extraSize = selectedSize - len(selectedIdx)
        selectedIdx.extend(spare[:extraSize])
    return selectedIdx


def exp_my2(target):
    killNum, mutate_predict_all = get_finetune_info(target, dataset, network)
    rank_lst = np.argsort(-killNum[shuffle_indices][:all_sample_size])
    return rank_lst


def exp_deepest(target, num):

    origin_data = np.array(pd.read_csv(f'data/deepest_info/{dataset}_{network}_{target}.csv'))
    chosen_data = origin_data[shuffle_indices][:all_sample_size]
    df = pd.DataFrame(np.array(chosen_data))
    df.to_csv(os.path.join('data/deepest_info', f"{dataset}_{network}_{target}_retrain.csv"), index=False)

    res_combo = os.popen(f"java -jar deepest.jar data/deepest_info/{dataset}_{network}_{target}_retrain.csv combo 0.70 {num}")
    text_combo = res_combo.read()

    line = text_combo.split("\n")[2]
    info = line.split(':')[-1].strip()[1:-1]
    selectedIdx = []
    for i in info.split(','):
        selectedIdx.append(int(i))

    res_combo.close()
    return selectedIdx


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset",
                        type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network",
                        type=str, default="lenet1")
    parser.add_argument("--ratio", "-r", type=float, default=0.5)
    args = parser.parse_args()

    dataset = args.dataset
    network = args.network

    global model, x_test, y_test, y_test_psedu, shuffle_indices, all_sample_size, \
        x_val, y_val, x_train, y_train
    save_path = f"results/retrain"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape = dataset_shape(name=dataset)
    x_train, y_train = get_train_general(name=dataset, network=network, shape=shape)
    shuffle_train_indices = np.random.permutation(np.arange(len(x_train)))
    x_train, y_train = x_train[shuffle_train_indices], y_train[shuffle_train_indices]


    data, index = [], []
    columns = [1000]
    baselines = ['nac', 'nac2', 'nbc', 'snac', 'tknc', 'lsa', 'dsa', 'deepgini', 'ats', 'deepest', 'random', 'mutate', 'my*', 'my']

    for target in ['cw', 'bim', 'fgsm', 'jsma']:
    # for target in ['cw']:

        index.extend([target+'_'+m for m in baselines])
        model = load_model_again(dataset, network)
        x_dau, y_dau = get_candidate_general(target, dataset, network)
        shuffle_indices = np.random.permutation(np.arange(len(x_dau)))
        x_dau, y_dau = x_dau[shuffle_indices], y_dau[shuffle_indices]
        dau_size = x_dau.shape[0]

        all_sample_size = int(dau_size*args.ratio)
        x_test, y_test = x_dau[:all_sample_size], y_dau[:all_sample_size]
        x_val, y_val = x_dau[all_sample_size:], y_dau[all_sample_size:]

        y_test_psedu = get_psedu_label(model, x_test)
        acc_val0 = np.sum(y_test == y_test_psedu)/all_sample_size
        y_test = keras.utils.to_categorical(y_test, 10)
        y_val = keras.utils.to_categorical(y_val, 10)

        rank_ats = exp_ats()
        rank_nac, rank_nac2, rank_nbc, rank_snac, rank_tknc = exp_coverage()
        rank_lsa, rank_dsa = exp_sa(target)
        rank_mutate = exp_mutate(target)
        rank_gini = exp_gini()
        rank_my2 = exp_my2(target)
        rank_random = np.arange(all_sample_size)
        np.random.shuffle(rank_random)

        class_Dic = {}
        for i, label in enumerate(y_test_psedu):
            if label not in class_Dic:
                class_Dic[label] = []
            class_Dic[label].append(i)

        acc_nac, acc_nbc, acc_snac, acc_tknc = [], [], [], []
        acc_lsa, acc_dsa, acc_ats, acc_random = [], [], [], []
        acc_deepest, acc_mutate, acc_my2, acc_my = [], [], [], []
        acc_gini, acc_nac2 = [], []

        for num in columns:

            filepath = f"retrained_model/{dataset}_{network}_nac_{num}.h5"
            acc_val = retrain(model, x_test[rank_nac][:num], y_test[rank_nac][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nac.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_nac2_{num}.h5"
            acc_val = retrain(model, x_test[rank_nac2][:num], y_test[rank_nac2][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nac2.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_nbc_{num}.h5"
            acc_val = retrain(model, x_test[rank_nbc][:num], y_test[rank_nbc][:num], filepath)
            model = load_model_again(dataset, network)
            acc_nbc.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_snac_{num}.h5"
            acc_val = retrain(model, x_test[rank_snac][:num], y_test[rank_snac][:num], filepath)
            model = load_model_again(dataset, network)
            acc_snac.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_tknc_{num}.h5"
            acc_val = retrain(model, x_test[rank_tknc][:num], y_test[rank_tknc][:num], filepath)
            model = load_model_again(dataset, network)
            acc_tknc.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_lsa_{num}.h5"
            acc_val = retrain(model, x_test[rank_lsa][:num], y_test[rank_lsa][:num], filepath)
            model = load_model_again(dataset, network)
            acc_lsa.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_dsa_{num}.h5"
            acc_val = retrain(model, x_test[rank_dsa][:num], y_test[rank_dsa][:num], filepath)
            model = load_model_again(dataset, network)
            acc_dsa.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_gini_{num}.h5"
            acc_val = retrain(model, x_test[rank_gini][:num], y_test[rank_gini][:num], filepath)
            model = load_model_again(dataset, network)
            acc_gini.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_ats_{num}.h5"
            acc_val = retrain(model, x_test[rank_ats][:num], y_test[rank_ats][:num], filepath)
            model = load_model_again(dataset, network)
            acc_ats.append(round(acc_val - acc_val0, 4) * 100)

            selectedIdx_deepest = exp_deepest(target, num)
            filepath = f"retrained_model/{dataset}_{network}_deepest_{num}.h5"
            acc_val = retrain(model, x_test[selectedIdx_deepest], y_test[selectedIdx_deepest], filepath)
            model = load_model_again(dataset, network)
            acc_deepest.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_random_{num}.h5"
            acc_val = retrain(model, x_test[rank_random][:num], y_test[rank_random][:num], filepath)
            model = load_model_again(dataset, network)
            acc_random.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_mutate_{num}.h5"
            acc_val = retrain(model, x_test[rank_mutate[:num]], y_test[rank_mutate[:num]], filepath)
            model = load_model_again(dataset, network)
            acc_mutate.append(round(acc_val - acc_val0, 4) * 100)

            filepath = f"retrained_model/{dataset}_{network}_my2_{num}.h5"
            acc_val = retrain(model, x_test[rank_my2][:num], y_test[rank_my2][:num], filepath)
            model = load_model_again(dataset, network)
            acc_my2.append(round(acc_val - acc_val0, 4) * 100)

            selectedIdx_my = exp_my(target, class_Dic, num)
            filepath = f"retrained_model/{dataset}_{network}_my_{num}.h5"
            acc_val = retrain(model, x_test[selectedIdx_my], y_test[selectedIdx_my], filepath)
            model = load_model_again(dataset, network)
            acc_my.append(round(acc_val - acc_val0, 4) * 100)

        data.append(acc_nac)
        data.append(acc_nac2)
        data.append(acc_nbc)
        data.append(acc_snac)
        data.append(acc_tknc)
        data.append(acc_lsa)
        data.append(acc_dsa)
        data.append(acc_gini)
        data.append(acc_ats)
        data.append(acc_deepest)
        data.append(acc_random)
        data.append(acc_mutate)
        data.append(acc_my2)
        data.append(acc_my)

    df = pd.DataFrame(np.array(data), columns=columns, index=index)
    df.to_csv(os.path.join(save_path, f"{dataset}_{network}_retrain_{args.ratio}.csv"))
>>>>>>> 1dffd00419da92d924b616008c876798ac08764e
