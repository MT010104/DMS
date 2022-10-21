import os
import keras
from keras.optimizers import SGD, adadelta

def load_model(dataset, network):
    model_name = f"{dataset}_{network}.h5"
    if os.path.exists(os.path.join("data/models", model_name)):
        print(os.path.join("data/models", model_name))
        model = keras.models.load_model(os.path.join("data/models", model_name))
    else:
        raise NotImplementedError(f"No such model for {model_name}")

    if network == 'vgg16':
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['accuracy'])
    return model

def recompile(network, model):
    if network == 'vgg16':
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta', metrics=['accuracy'])
    return model