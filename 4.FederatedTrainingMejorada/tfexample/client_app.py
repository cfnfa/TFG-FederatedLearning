"""tfexample: A Flower / TensorFlow app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tfexample.task import load_data, load_model


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        self.model = load_model(sequence_length=6, input_dim=4, learning_rate=learning_rate)
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, rmse = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        return loss, len(self.x_test), {"rmse": rmse}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    node_id = context.node_id 
     # `partition-id` uniquely identifies the client
    X_train, X_test, y_train, y_test, archivo, scaler = load_data(partition_id)
    print(f"Configuración del nodo creada: {context.node_config}")
    print(f'Este cliente tiene el partition_id {partition_id} ,el node_id {node_id} y el archivo {archivo}')

    data= X_train, X_test, y_train, y_test

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
