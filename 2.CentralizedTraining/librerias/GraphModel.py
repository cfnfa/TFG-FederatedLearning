import tensorflow.keras.regularizers
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Conv1D, MaxPool1D, Flatten, concatenate, Reshape
from tensorflow.keras import Model


class GraphLSTM:
    def __init__(self, input_shape=(12, 1), output_shape=1, hidden_units=128, hidden_units_2=12, hidden_units_3=64,
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

    def build(self):
        i_g = Input(shape=self.input_shape, name="Glucose")
        i_h = Input(shape=self.input_shape, name="Heart Rate")
        i_i = Input(shape=(12,), name="Insulin")
        i_c = Input(shape=(12,), name="Carbohydrates")

        x_g = LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_Glu",
                   return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i_g)
        x_h = LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_HR",
                   return_sequences=True, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i_h)
        x_g = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_Glu_2",
                   return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x_g)
        x_h = LSTM(self.hidden_units_2, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_HR_2",
                   return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x_h)
        '''x_i = LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_Ins",
                   return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i_i)
        x_c = LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, name="LSTM_Carb",
                   return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(i_c)'''
        x_g = Reshape((12, 1))(x_g)
        x_h = Reshape((12, 1))(x_h)
        x_i = Reshape((12, 1))(i_i)
        x_c = Reshape((12, 1))(i_c)
        x = concatenate([x_g, x_h, x_i, x_c])
        x_h = LSTM(self.hidden_units_3, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                   name="LSTM",
                   return_sequences=False, kernel_regularizer=tensorflow.keras.regularizers.L2(0.001))(x_h)
        x = Dense(1, name="Prediction", activation=None)(x)

        model = Model(inputs=[i_g, i_h, i_i, i_c], outputs=[x])

        tensorflow.keras.utils.plot_model(model, 'model.png', show_shapes=True, show_layer_names=True)

        return model
