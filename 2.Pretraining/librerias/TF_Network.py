import tensorflow.keras.regularizers
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Conv1D, MaxPool1D, Flatten, BatchNormalization
from tensorflow.keras import Model


class TF_LSTM:
    def __init__(self, input_shape=(18, 3), output_shape=1, hidden_units=128, hidden_units_2=128, hidden_units_3=128,
                 layers=1, dropout=0.0, lr=0.001, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.layers = layers
        self.lr = lr
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'tseq_6_q_5_neurons_{}_neurons2_{}_neurons3_{}_lr_{}'.format(self.hidden_units, self.hidden_units_2,
                                                                            self.hidden_units_3, self.lr)

    def build(self):
        i = Input(shape=self.input_shape, name='Base_model')

        x = LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name='LSTM_1',
                 return_sequences=self.layers > 1, kernel_initializer='glorot_normal',
                 kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i)
        if self.layers > 1:
            for _ in range(self.layers - 2):
                x = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                         return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001),
                         name='LSTM_2')(x)
            x = LSTM(self.hidden_units_3, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name='LSTM_2',
                     return_sequences=False, kernel_initializer='glorot_normal',
                     kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)

        x = Dense(self.output_shape, activation=None, kernel_initializer='glorot_normal', name='Dense')(x)

        model = Model(inputs=[i], outputs=[x])

        '''i = Input(shape=self.input_shape)

        x = LSTM(64, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                 return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i)
        x = LSTM(128, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                 return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
        x = LSTM(256, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                 return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
        x = LSTM(128, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                 return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
        x = LSTM(64, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                 return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
        x = Dense(self.output_shape, activation=None)(x)'''

        return model


class LSTM_1D:
    def __init__(self, input_shape=(18, 3), output_shape=1, hidden_units=128, hidden_units_2=128, hidden_units_3=128,
                 layers=1, dropout=0.0, lr=0.01, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.layers = layers
        self.lr = lr
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'tseq_6_q_5_neurons_{}_neurons2_{}_neurons3_{}_lr_{}'.format(self.hidden_units, self.hidden_units_2,
                                                                            self.hidden_units_3, self.lr)

    def build(self):
        i = Input(shape=self.input_shape)
        x = Conv1D(self.hidden_units, 5, strides=1, padding='causal', activation='tanh')(i)
        if self.layers == 1:
            x = MaxPool1D(2)(x)
            x = Flatten()(x)
        if self.layers > 1:
            if self.layers > 2:
                x = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                         return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
            x = LSTM(self.hidden_units_3, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                     return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
        x = Dense(self.output_shape)(x)

        return Model(inputs=[i], outputs=[x])


class BI_LSTM:
    def __init__(self, input_shape=(18, 3), output_shape=1, hidden_units=128, hidden_units_2=128, hidden_units_3=128,
                 layers=1, dropout=0.0, lr=0.01, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.layers = layers
        self.lr = lr
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'tseq_6_q_5_neurons_128_neurons2_0_neurons3_0_lr_0.01'

    def build(self):
        i = Input(shape=self.input_shape)

        x = Bidirectional(LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                               return_sequences=self.layers > 1,
                               kernel_regularizer=tensorflow.keras.regularizers.L2(0.001)))(i)
        if self.layers > 1:
            for _ in range(self.layers - 2):
                x = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                         return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
            x = LSTM(self.hidden_units_3, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                     return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)

        x = Dense(self.output_shape, activation=None)(x)

        return Model(inputs=[i], outputs=[x])


class CNN_1D:
    def __init__(self, input_shape=(18, 3), output_shape=1, hidden_units=128, hidden_units_2=128, hidden_units_3=128,
                 layers=1, dropout=0.0, lr=0.01, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_units = hidden_units
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.layers = layers
        self.lr = lr
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'tseq_6_q_5_neurons_{}_neurons2_{}_neurons3_{}_lr_{}'.format(self.hidden_units, self.hidden_units_2,
                                                                            self.hidden_units_3, self.lr)

    def build(self):
        i = Input(shape=self.input_shape)

        x = Conv1D(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                   kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i)
        if self.layers > 1:
            for _ in range(self.layers - 2):
                x = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                         return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)
            x = LSTM(self.hidden_units_3, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                     return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x)

        x = Dense(self.output_shape, activation=None)(x)

        return Model(inputs=[i], outputs=[x])
