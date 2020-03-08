import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks

if int(tf.__version__[0]) > 1:
    session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=session_config)
else:
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_config)


def get_arch(arch):
    #  note that the 'net' and 'fir' part of the arch string is functional... (diff vs no diff)
    print('Grabbing {}'.format(arch))
    if arch == 'net_gru_000':
        raise Exception('Place holder.')
    elif arch == 'net_cug_000':
        model = Sequential()
        model.add(layers.CuDNNGRU(16, return_sequences=True, input_shape=(None, 1)))
        model.add(layers.CuDNNGRU(4, return_sequences=True))
        model.add(layers.Dense(2, activation='linear'))
        # note the loss!
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_cug_001':
        model = Sequential()
        model.add(layers.CuDNNGRU(16, return_sequences=True, input_shape=(None, 1)))
        model.add(layers.CuDNNGRU(16, return_sequences=True))
        model.add(layers.Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_cug_002':
        model = Sequential()
        model.add(layers.CuDNNGRU(16, return_sequences=True, input_shape=(None, 1)))
        model.add(layers.CuDNNGRU(16, return_sequences=True))
        model.add(layers.Dense(16))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(2, activation='linear'))
        # note the loss!
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_cug_003':
        model = Sequential()
        model.add(layers.CuDNNGRU(4, return_sequences=True, input_shape=(None, 1)))
        model.add(layers.Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_cug_004':
        model = Sequential()
        model.add(layers.CuDNNGRU(2, return_sequences=True, input_shape=(None, 1)))
        model.add(layers.Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_sgd_000':
        model = Sequential()
        model.add(layers.Dense(2, activation='linear', input_dim=250))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_sgd_001':
        input1 = layers.Input(shape=(250,))
        x1 = layers.Dense(15, activation='linear')(input1)
        x1nl = layers.ReLU()(x1)
        x2 = layers.Dense(15, activation='softmax')(x1nl)
        mult1 = layers.Multiply()([x1, x2])
        mult1d = layers.Dropout(0.3)(mult1)
        out = layers.Dense(2)(mult1d)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_sgd_002':
        input1 = layers.Input(shape=(250,))
        x1 = layers.Dense(15, activation='linear')(input1)
        x1nl = layers.ReLU()(x1)
        x2 = layers.Dense(15, activation='softmax')(x1nl)
        mult1 = layers.Multiply()([x1, x2])
        mult1d = layers.Dropout(0.4)(mult1)
        out = layers.Dense(2)(mult1d)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif arch == 'net_sgd_003':
        input1 = layers.Input(shape=(250,))
        x1 = layers.Dense(15, activation='linear')(input1)
        x1nl = layers.ReLU()(x1)
        x2 = layers.Dense(15, activation='softmax')(x1nl)
        mult1 = layers.Multiply()([x1, x2])
        mult1d = layers.Dropout(0.5)(mult1)
        out = layers.Dense(2)(mult1d)
        model = Model(inputs=input1, outputs=out)
        model.compile(loss='mean_squared_error', optimizer='adam')

    else:
        raise Exception("Undefined network arch: {}".format(arch))

    return model
