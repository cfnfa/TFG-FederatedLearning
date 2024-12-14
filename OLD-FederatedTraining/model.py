import tensorflow as tf

def load_lstm_model(sequence_length: int = 12, input_dim: int = 4):
    """Create and compile the LSTM model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=(sequence_length, input_dim), return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))  # Single output for regression

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='mean_squared_error')
    return model

