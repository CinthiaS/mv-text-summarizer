from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Dropout, InputLayer
from keras.models import Sequential


def simple_nn(metrics):

    model = Sequential()
    model.add(InputLayer(input_shape=(14,)))
    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=metrics)

    return model