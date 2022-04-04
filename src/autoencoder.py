from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, concatenate, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


def save_model(model, bottleneck_dim, section, path_to_write):
    
    model_json = model.to_json()
    with open('{}/autoencoder_{}.json'.format(path_to_write, section), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('{}/autoencoder_{}.h5'.format(path_to_write, section))
    print("Saved model to disk")

def fit_autoencoder(X_embedd, X_features, y_train, bottleneck_dim, section, path_to_write):
    
    sequence_input = Input(shape=(X_embedd.shape[1],), dtype='int32')

    e_1 = Dense(X_embedd.shape[1]*2)(sequence_input)
    e_1 = BatchNormalization()(e_1)
    e_1 = LeakyReLU()(e_1)

    e_2 = Dense(X_embedd.shape[1])(e_1)
    e_2 = BatchNormalization()(e_2)
    e_2 = LeakyReLU()(e_2)


    #encoder vision 2
    sequence_input2 = Input(shape=(X_features.shape[1],), dtype='int32')

    e_3 = Dense(X_features.shape[1]*2)(sequence_input2)
    e_3 = BatchNormalization()(e_3)
    e_3 = LeakyReLU()(e_3)

    e_4 = Dense(X_features.shape[1]*2)(sequence_input2)
    e_4 = BatchNormalization()(e_4)
    e_4 = LeakyReLU()(e_4)

    #Concatenate visions
    v_1 = e_2

    v_2_concat = concatenate([v_1, e_3])
    v_2 = Dense(256, activation='relu')(v_2_concat)

    v_3_concat = concatenate([v_1, v_2, e_4])
    v_3 = Dense(256, activation='relu')(v_3_concat)

    out_concat = concatenate([v_1, v_2, v_3])

    #Shared Inputs

    shared_input = Dense(bottleneck_dim)(out_concat)
    bottleneck = Dense(bottleneck_dim)(shared_input)

    # decoder  vision 1
    d_1 = Dense(X_embedd.shape[1])(bottleneck)
    d_1 = BatchNormalization()(d_1)
    d_1 = LeakyReLU()(d_1)
    dropout1 = Dropout(.2)(d_1)

    d_2 = Dense(X_embedd.shape[1])(dropout1)
    d_2 = BatchNormalization()(d_2)
    d_2 = LeakyReLU()(d_2)
    dropout2 = Dropout(.2)(d_2)

    d_v1 = Dense(X_embedd.shape[1])(dropout2)
    d_v1 = BatchNormalization()(d_v1)
    d_v1 = LeakyReLU()(d_v1)

    #decoder vision 2
    d_5 = Dense(X_features.shape[1])(bottleneck)
    d_5 = BatchNormalization()(d_5)
    d_5 = LeakyReLU()(d_5)
    dropout3 = Dropout(.2)(d_5)

    d_4 = Dense(X_embedd.shape[1])(dropout3)
    d_4 = BatchNormalization()(d_4)
    d_4 = LeakyReLU()(d_4)
    dropout4 = Dropout(.2)(d_4)

    d_v2 = Dense(X_features.shape[1])(dropout4)
    d_v2 = BatchNormalization()(d_v2)
    d_v2 = LeakyReLU()(d_v2)

    output_v1 = Dense(X_embedd.shape[1], activation='linear')(d_v1)
    output_v2 = Dense(X_features.shape[1], activation='linear')(d_v2)

    model = Model(inputs=[sequence_input, sequence_input2], outputs=[output_v1, output_v2])

    model.compile(optimizer=keras.optimizers.Adam(
                    learning_rate=0.0001) ,loss=keras.metrics.mean_squared_error)
    
    one_hot_label = to_categorical(y_train)
    X_train_embedd, X_valid_embedd, y_train_embedd, y_valid_embedd = train_test_split(
            X_embedd, one_hot_label, stratify=one_hot_label, shuffle=True, test_size=0.2)

    one_hot_label = to_categorical(y_train)
    X_train_features, X_valid_features, y_train_features, y_valid_features = train_test_split(
            X_features, one_hot_label, stratify=one_hot_label, shuffle=True, test_size=0.2)
    
    history = model.fit(
            x=[X_train_embedd, X_train_features], y=[X_train_embedd, X_train_features],
                epochs=1, validation_data=([X_valid_embedd, X_valid_features], [X_valid_embedd, X_valid_features]),
            shuffle=True, batch_size=64)
        
    encoder = Model(inputs=[sequence_input, sequence_input2], outputs=bottleneck)
    encoder.save('{}/encoder_{}.h5'.format(path_to_write, section))
    
    save_model(model, bottleneck_dim, section, path_to_write)
    
    return model

def create_representation(
    dataset, sections, path_to_write, verbose=True):
    
    for section in sections:
    
        X_features_test = dataset[section]["X_train_features"]
        X_features_train = dataset[section]["y_train"]
    
        X_embedd_test = dataset['X_test_embbed']
        X_embedd_train = dataset['X_train_embbed']

        encoder = load_model('{}/encoder_{}.h5'.format(path_to_write, section))

        X_test_encode = encoder.predict([X_embedd_test, X_features_test])
        X_train_encode = encoder.predict([X_embedd_train, X_features_train])

        dataset[section]['X_train_f1'] = X_train_encode
        dataset[section]['X_train_f1'] = X_test_encode

    if verbose:
        print("Write dataset")
    
        with open('{}/dataset_{}.pkl'.format(path_to_write, 'features'), 'wb') as fp:
            pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

def main_autoencoder(
    dataset, sections, X1, X2, y, path_to_read='dataset', path_to_write='../autoencoder_test', bottleneck_dim=64):
    
    
    columns = [str(i) for i in range(300)]
    
    for section in sections:
        
        X_features = dataset[section][X1]
        y_train = dataset[section][y]

        X_embedd = dataset[section][X2]
        
        model = fit_autoencoder(X_embedd, X_features, y_train, bottleneck_dim, section, path_to_write)
        
    create_representation(dataset, sections, path_to_write, verbose=True)