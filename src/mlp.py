from tensorflow import keras
import math
import pickle

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import metrics


import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def plot_history(history,dataset):

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loos x Epoch - {}'.format(dataset))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    return plt

def simple_nn(input_shape=14, epochs=100, learning_rate=0.001):

    m = [
        metrics.MeanSquaredError(name="mean_squared_error"),
        metrics.MeanAbsoluteError(name="mean_absolute_error")]

    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu',kernel_initializer=keras.initializers.GlorotUniform()))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='relu', kernel_initializer=keras.initializers.GlorotUniform()))


    model.compile(loss='mae', optimizer=keras.optimizers.Adam(
        learning_rate=learning_rate, clipnorm=1.0), metrics=m)

    return model

def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


def train_neural_model(dataset, view, X_train, y_train, input_shape=14, validation_split=0.2, learning_rate=0.001, epochs=100, batch_size=64, verbose=0):

    model = simple_nn(epochs=1, input_shape=input_shape, learning_rate=learning_rate)

    lrate = LearningRateScheduler(step_decay)
    early = EarlyStopping(monitor='loss', patience=20)
    callbacks_list = [lrate, early]
    
    history = model.fit(
        X_train, y_train, validation_split=validation_split, epochs=epochs,
         batch_size=batch_size, verbose=verbose,  callbacks=callbacks_list)

    with open('history_{}_{}'.format(dataset, view), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model_json = model.to_json()
    with open("model_{}_{}.json".format(dataset, view), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_{}_{}.h5".format(dataset,view))
    print("Saved model to disk")

def pipeline_neural_model(
    dataset, view,  X_train, y_train, X_test, y_test, input_shape=14, validation_split=0.2, learning_rate=0.001,
     epochs=100, batch_size=64, verbose=0, train=False):

    if train:
        train_neural_model(dataset, view, X_train, y_train, input_shape=input_shape, validation_split=validation_split, learning_rate=learning_rate,
         epochs=epochs, batch_size=batch_size, verbose=verbose)

    json_file = open('model_{}_{}.json'.format(dataset, view), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('model_{}_{}.h5'.format(dataset, view))

    infile = open('history_{}_{}'.format(dataset, view),'rb')
    history = pickle.load(infile)
    infile.close()

    y_pred = model.predict(X_test)

    plt = plot_history(history, dataset)
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae