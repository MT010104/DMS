import os
import argparse
import sys
import keras
import numpy as np
import pandas as pd

from ATS.ATS import ATS
import metrics

from tools.data_utils import dataset_shape, get_sa_general
from tools.data_utils import get_candidate_general, get_train_general
from tools.data_utils import get_finetune_info, get_mutate_info
from tools.model_utils import load_model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def diverse_errors_num(y_s, y_psedu):
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)
    return len(fault_idx_arr)


def fault_detection(y, y_psedu):
    fault_num = np.sum(y != y_psedu)
    print("fault num : {}".format(fault_num))

    diverse_fault_num = diverse_errors_num(y, y_psedu)
    print("diverse_fault_num  : {}/{}".format(diverse_fault_num, 90))
    return fault_num, diverse_fault_num


def exp_ats():
    ats = ATS()
    nb_classes = 10
    rank_lst, _, _ = ats.get_priority_sequence(x_test, y_test_psedu, nb_classes, model, th=0.001)
    return rank_lst


def exp_gini():
    pred_test_prob = model.predict(x_test)
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
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

    shape = dataset_shape(name=dataset)
    x_train, _ = get_train_general(name=dataset, network=network, shape=shape)

    ac = metrics.nac(x_test, input, layers, t=0.5)
    rank_ac = ac.rank_2(x_test)
    rank_ac2 = ac.rank_fast(x_test)

    bc = metrics.nbc(x_train, input, layers, std=std)
    rank_bc = bc.rank_2(x_test, use_lower=True)

    rank_snac = bc.rank_2(x_test, use_lower=False)

    tk = metrics.tknc(x_test, input, layers, k=3)
    rank_tk = tk.rank(x_test)

    return rank_ac, rank_ac2, rank_bc, rank_snac, rank_tk


def exp_sa(target):
    lsa, dsa = get_sa_general(target, dataset, network)
    rank_lsa = np.argsort(-lsa)
    rank_dsa = np.argsort(-dsa)
    return rank_lsa, rank_dsa


def exp_mutate(target):
    killNum, _ = get_mutate_info(target, dataset, network)
    return np.argsort(-killNum)


def exp_my(target, class_Dic, selectedSize):

    killNum, finetune_predict_all = get_finetune_info(target, dataset, network)
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
            # subSelectedSize = int(np.around(len(subclass_Dic[label2]) / len(candidates) * selectedSizeFromClass))
            selectedIdx.extend(subclass_Dic[label2][:subSelectedSize])
            if subSelectedSize < len(subclass_Dic[label2]):
                spare.extend(subclass_Dic[label2][subSelectedSize:])

    spare = np.array(spare)
    spare = spare[np.argsort(-killNum[spare])]
    if len(selectedIdx) < selectedSize:
        extraSize = selectedSize - len(selectedIdx)
        selectedIdx.extend(spare[:extraSize])
    elif len(selectedIdx) > selectedSize:
        print(len(selectedIdx), selectedSize)
        extraSize = len(selectedIdx) - selectedSize
        remove_lst = np.array(selectedIdx)[np.argsort(killNum[selectedIdx])[:extraSize]]
        for idx in remove_lst:
            selectedIdx.remove(idx)
    return selectedIdx


def exp_my2(target):
    killNum, _ = get_finetune_info(target, dataset, network)
    rank_lst = np.argsort(-killNum)
    return rank_lst

def exp_deepest(target, num):

    res_combo = os.popen(f"java -jar deepest.jar data/deepest_info/{dataset}_{network}_{target}.csv combo 0.70 {num}")
    text_combo = res_combo.read()

    line = text_combo.split("\n")[1]
    info = line.split(':')[-1].strip()[1:-1]
    selectedIdx = []
    for i in info.split(','):
        selectedIdx.append(int(i))

    res_combo.close()
    return selectedIdx


if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network", type=str, default="lenet5")
    parser.add_argument("--rq", "-rq", help="fault or diversity", type=str, default="fault")
    args = parser.parse_args()

    dataset = args.dataset
    network = args.network

    global model, x_test, y_test, y_test_psedu
    save_path = f"results/{args.rq}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data, index = [], []
    last_column_fault, last_column_diversity = [], []
    if args.rq == 'fault':
        columns = list(np.arange(100, 1001, 100))
    else:
        columns = list(np.arange(50, 501, 50))

    baselines = ['nac', 'nac2', 'nbc', 'snac', 'tknc', 'lsa', 'dsa', 'gini', 'ats', 'deepest', 'random', 'mutate', 'my*', 'my']

    for target in ['nature', 'cw', 'bim', 'fgsm', 'jsma']:
        index.extend([target+'_'+m for m in baselines])
        model = load_model(dataset, network)
        x_test, y_test = get_candidate_general(target, dataset, network)
        all_sample_size = x_test.shape[0]
        y_test_psedu = get_psedu_label(model, x_test)
        fault_num_all, diverse_num_all = fault_detection(y_test, y_test_psedu)
        last_column_fault.extend(np.tile(fault_num_all, len(baselines)))
        last_column_diversity.extend(np.tile(diverse_num_all, len(baselines)))

        rank_ats = exp_ats()
        rank_nac, rank_nac2, rank_nbc, rank_snac, rank_tknc = exp_coverage()
        rank_lsa, rank_dsa = exp_sa(target)
        rank_my2 = exp_my2(target)
        rank_gini = exp_gini()
        rank_random = np.arange(all_sample_size)
        rank_mutate = exp_mutate(target)
        np.random.shuffle(rank_random)

        bug_detection_nac, bug_diversity_nac = [], []
        bug_detection_nac2, bug_diversity_nac2 = [], []
        bug_detection_nbc, bug_diversity_nbc = [], []
        bug_detection_snac, bug_diversity_snac = [], []
        bug_detection_tknc, bug_diversity_tknc = [], []
        bug_detection_lsa, bug_diversity_lsa = [], []
        bug_detection_dsa, bug_diversity_dsa = [], []
        bug_detection_ats, bug_diversity_ats = [], []
        bug_detection_gini, bug_diversity_gini = [], []
        bug_detection_random, bug_diversity_random = [], []
        bug_detection_deepest, bug_diversity_deepest = [], []
        bug_detection_mutate, bug_diversity_mutate = [], []
        bug_detection_my2, bug_diversity_my2 = [], []
        bug_detection_my, bug_diversity_my = [], []

        all_sample_size = x_test.shape[0]

        class_Dic = {}
        for i, label in enumerate(y_test_psedu):
            if label not in class_Dic:
                class_Dic[label] = []
            class_Dic[label].append(i)

        for num in columns:
            
            #nac, nbc, snac, tknc, lsa, dsa, ats
            fault_num_nac, diverse_num_nac = fault_detection(y_test[rank_nac][:num], y_test_psedu[rank_nac][:num])
            bug_detection_nac.append(np.around(fault_num_nac/fault_num_all*100, 2))
            bug_diversity_nac.append(diverse_num_nac)

            fault_num_nac2, diverse_num_nac2 = fault_detection(y_test[rank_nac2][:num], y_test_psedu[rank_nac2][:num])
            bug_detection_nac2.append(np.around(fault_num_nac2 / fault_num_all * 100, 2))
            bug_diversity_nac2.append(diverse_num_nac2)

            fault_num_nbc, diverse_num_nbc = fault_detection(y_test[rank_nbc][:num], y_test_psedu[rank_nbc][:num])
            bug_detection_nbc.append(np.around(fault_num_nbc / fault_num_all * 100, 2))
            bug_diversity_nbc.append(diverse_num_nbc)

            fault_num_snac, diverse_num_snac = fault_detection(y_test[rank_snac][:num], y_test_psedu[rank_snac][:num])
            bug_detection_snac.append(np.around(fault_num_snac / fault_num_all * 100, 2))
            bug_diversity_snac.append(diverse_num_snac)

            fault_num_tknc, diverse_num_tknc = fault_detection(y_test[rank_tknc][:num], y_test_psedu[rank_tknc][:num])
            bug_detection_tknc.append(np.around(fault_num_tknc / fault_num_all * 100, 2))
            bug_diversity_tknc.append(diverse_num_tknc)

            fault_num_lsa, diverse_num_lsa = fault_detection(y_test[rank_lsa][:num], y_test_psedu[rank_lsa][:num])
            bug_detection_lsa.append(np.around(fault_num_lsa / fault_num_all * 100, 2))
            bug_diversity_lsa.append(diverse_num_lsa)

            fault_num_dsa, diverse_num_dsa = fault_detection(y_test[rank_dsa][:num], y_test_psedu[rank_dsa][:num])
            bug_detection_dsa.append(np.around(fault_num_dsa / fault_num_all * 100, 2))
            bug_diversity_dsa.append(diverse_num_dsa)

            fault_num_gini, diverse_num_gini = fault_detection(y_test[rank_gini][:num], y_test_psedu[rank_gini][:num])
            bug_detection_gini.append(np.around(fault_num_gini / fault_num_all * 100, 2))
            bug_diversity_gini.append(diverse_num_gini)

            fault_num_ats, diverse_num_ats = fault_detection(y_test[rank_ats][:num], y_test_psedu[rank_ats][:num])
            bug_detection_ats.append(np.around(fault_num_ats / fault_num_all * 100, 2))
            bug_diversity_ats.append(diverse_num_ats)

            selectedIdx_deepest = exp_deepest(target, num)
            fault_num_deepest, diverse_num_deepest = fault_detection(y_test[selectedIdx_deepest],
                                                                     y_test_psedu[selectedIdx_deepest])
            bug_detection_deepest.append(np.around(fault_num_deepest / fault_num_all * 100, 2))
            bug_diversity_deepest.append(diverse_num_deepest)

            fault_num_random, diverse_num_random = fault_detection(y_test[rank_random][:num], y_test_psedu[rank_random][:num])
            bug_detection_random.append(np.around(fault_num_random / fault_num_all * 100, 2))
            bug_diversity_random.append(diverse_num_random)

            fault_num_mutate, diverse_num_mutate = fault_detection(y_test[rank_mutate[:num]], y_test_psedu[rank_mutate[:num]])
            bug_detection_mutate.append(np.around(fault_num_mutate / fault_num_all * 100, 2))
            bug_diversity_mutate.append(diverse_num_mutate)

            fault_num_my2, diverse_num_my2 = fault_detection(y_test[rank_my2][:num], y_test_psedu[rank_my2][:num])
            bug_detection_my2.append(np.around(fault_num_my2 / fault_num_all * 100, 2))
            bug_diversity_my2.append(diverse_num_my2)

            selectedIdx_my = exp_my(target, class_Dic, num)
            fault_num_my, diverse_num_my = fault_detection(y_test[selectedIdx_my], y_test_psedu[selectedIdx_my])
            bug_detection_my.append(np.around(fault_num_my / fault_num_all * 100, 2))
            bug_diversity_my.append(diverse_num_my)


        if args.rq == 'fault':
            data.append(bug_detection_nac)
            data.append(bug_detection_nac2)
            data.append(bug_detection_nbc)
            data.append(bug_detection_snac)
            data.append(bug_detection_tknc)
            data.append(bug_detection_lsa)
            data.append(bug_detection_dsa)
            data.append(bug_detection_gini)
            data.append(bug_detection_ats)
            data.append(bug_detection_deepest)
            data.append(bug_detection_random)
            data.append(bug_detection_mutate)
            data.append(bug_detection_my2)
            data.append(bug_detection_my)
        else:
            data.append(bug_diversity_nac)
            data.append(bug_diversity_nac2)
            data.append(bug_diversity_nbc)
            data.append(bug_diversity_snac)
            data.append(bug_diversity_tknc)
            data.append(bug_diversity_lsa)
            data.append(bug_diversity_dsa)
            data.append(bug_diversity_gini)
            data.append(bug_diversity_ats)
            data.append(bug_diversity_deepest)
            data.append(bug_diversity_random)
            data.append(bug_diversity_mutate)
            data.append(bug_diversity_my2)
            data.append(bug_diversity_my)
    if args.rq == 'fault':
        data = np.hstack((np.array(data), np.expand_dims(last_column_fault, axis=1)))
    else:
        data = np.hstack((np.array(data), np.expand_dims(last_column_diversity, axis=1)))

    columns.append('all')
    df = pd.DataFrame(np.array(data), columns=columns, index=index)
    df.to_csv(os.path.join(save_path, f"{dataset}_{network}_{args.rq}.csv"))
