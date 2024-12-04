from collections import OrderedDict
import tensorflow as tf
from omegaconf import DictConfig

from model import load_lstm_model
from dataset import prepare_test_set

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(sequence_length: int, input_dim: int):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        # Reconstruct the model from the weights received from the clients
        model = load_lstm_model(sequence_length, input_dim)
        print(f"Pesos recibidos por el servidor: {len(parameters)} capas.")

        # Convertimos los parámetros a un formato que pueda usar TensorFlow.
        weights = [tf.convert_to_tensor(w) for w in parameters]
        model.set_weights(weights)  # Asignar los pesos directamente

        # Compile the model before evaluating (as TensorFlow models need compilation)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        # Evaluate the model on the validation/test dataset
        x_test, y_test = prepare_test_set()
        val_loss, val_mae = model.evaluate(x_test, y_test, verbose=0)

        # Return loss and any additional metrics (in this case Mean Absolute Error)
        return val_loss, {"mae": val_mae}

    return evaluate_fn




