import keras
import os
import numpy as np
from tqdm import tqdm
import argparse
from tools.data_utils import get_train_general
from tools.model_utils import load_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset",
                        type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network",
                        type=str, default="lenet5")
    args = parser.parse_args()
    dataset = args.dataset
    network = args.network

    # Load models and datasets
    model = load_model(dataset, network)
    x_train, y_train = get_train_general(dataset, network)
    Y_train = np.argmax(y_train, axis=1)
    y_predict = np.argmax(model.predict(x_train, verbose=0), axis=1)
    origin_acc = model.evaluate(x_train, y_train, verbose=0)[1]
    print(f"origin_acc : {origin_acc}")

    x_true = []
    x_false = []
    for i, label in enumerate(y_predict):
        if label == Y_train[i]:
            x_true.append(i)
        else:
            x_false.append(i)

    batch_size = 32
    nb_epoch = 25
    num_classes = 10

    for iter in range(40):
        save_dir = f'finetuned_models/{dataset}_{network}/iter_{iter}_models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        y_predict[x_false] = np.random.randint(0, 10, size=len(x_false))
        y_predict_categorical = keras.utils.to_categorical(y_predict, num_classes=num_classes)
        checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                       verbose=1, save_weights_only=False, period=1)
        model.fit(x_train, y_predict_categorical, verbose=1,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, callbacks=[checkpointer])