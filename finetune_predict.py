<<<<<<< HEAD
import numpy.core.defchararray
from keras import Model
import keras
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K

import tools.data_utils as DataUtils
import tools.model_utils as ModelUtils

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset",
                        type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network",
                        type=str, default="lenet5")
    args = parser.parse_args()

    dataset = args.dataset
    network = args.network

    targets = ["nature", "fgsm", "cw", "bim", "jsma"]
    for target in targets:
        if target == 'nature'
            shape = DataUtils.dataset_shape(name=dataset)
            inputs, labels = DataUtils.get_clean_general(dataset, network, shape)
        else:
            inputs, labels = DataUtils.get_adv_general(target, dataset, network)


        all_sample_size = inputs.shape[0]
        model = ModelUtils.load_model(network=network, dataset=dataset)
        y_predict = np.argmax(model.predict(inputs, verbose=0), axis=1)

        for iter in range(40):

            model_dir = f'finetuned_models/{dataset}_{network}/iter_{iter}_models'
            save_dir = f'finetuned_prediction/{dataset}_{network}/iter_{iter}_models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            prediction = []
            finetuned_models = [f'model_{str(num).zfill(3)}.h5' for num in range(1, 51)]
            for model_name in tqdm(finetuned_models):
                K.clear_session()
                tf.reset_default_graph()
                model = keras.models.load_model(f"{model_dir}/{model_name}")
                single_mutate_prediction = np.argmax(model.predict(inputs, verbose=0), axis=1)
                prediction.append(single_mutate_prediction)
            print(np.array(prediction).shape)
=======
import numpy.core.defchararray
from keras import Model
import keras
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K

import tools.data_utils as DataUtils
import tools.model_utils as ModelUtils

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset",
                        type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network",
                        type=str, default="lenet5")
    args = parser.parse_args()

    dataset = args.dataset
    network = args.network

    targets = ["nature", "fgsm", "cw", "bim", "jsma"]
    for target in targets:
        if target == 'nature'
            shape = DataUtils.dataset_shape(name=dataset)
            inputs, labels = DataUtils.get_clean_general(dataset, network, shape)
        else:
            inputs, labels = DataUtils.get_adv_general(target, dataset, network)


        all_sample_size = inputs.shape[0]
        model = ModelUtils.load_model(network=network, dataset=dataset)
        y_predict = np.argmax(model.predict(inputs, verbose=0), axis=1)

        for iter in range(40):

            model_dir = f'finetuned_models/{dataset}_{network}/iter_{iter}_models'
            save_dir = f'finetuned_prediction/{dataset}_{network}/iter_{iter}_models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            prediction = []
            finetuned_models = [f'model_{str(num).zfill(3)}.h5' for num in range(1, 51)]
            for model_name in tqdm(finetuned_models):
                K.clear_session()
                tf.reset_default_graph()
                model = keras.models.load_model(f"{model_dir}/{model_name}")
                single_mutate_prediction = np.argmax(model.predict(inputs, verbose=0), axis=1)
                prediction.append(single_mutate_prediction)
            print(np.array(prediction).shape)
>>>>>>> 1dffd00419da92d924b616008c876798ac08764e
            np.save(f"{save_dir}/prediction_target_{target}.npy", np.array(prediction))