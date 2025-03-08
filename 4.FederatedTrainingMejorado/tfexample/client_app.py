"""Author: Clara Fuertes Novillo"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tfexample.task import load_data_crosssilo, load_model, clarke_error_grid, zone_percentages_new
from scipy.stats import pearsonr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


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
        self.model = load_model(sequence_length=12, input_dim=4, learning_rate=learning_rate)
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
        loss, rmse = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, verbose=2)
        y_pred = self.model.predict(self.x_test, verbose=False).flatten()

        global_min=np.array([0,0,39,0])
        global_max=np.array([25,1.01325, 400, 200])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.min_ = -global_min / (global_max - global_min)
        scaler.scale_ = 1 / (global_max - global_min)

        pearson_corr, _ = pearsonr(self.y_test.flatten(), y_pred)
        print(f"Correlación de Pearson: {pearson_corr}")

        # Desnormalizar los valores de referencia y las preds para calcular las zonas de Clarke
        y_test_zeros = np.zeros((len(self.y_test), 2))  
        y_pred_zeros = np.zeros((len(y_pred), 2))  

        # Reajustar las formas para asegurarse de que tienen dos dimensiones
        y_test_reshaped = self.y_test.reshape(-1, 1)  
        y_pred_reshaped = y_pred.reshape(-1, 1)  

        # Concatenar las matrices 
        y_test_desnormalized = scaler.inverse_transform(np.concatenate([y_test_zeros, y_test_reshaped, np.zeros((len(self.y_test), 1))], axis=1))[:, 2]
        y_pred_desnormalized = scaler.inverse_transform(np.concatenate([y_pred_zeros, y_pred_reshaped, np.zeros((len(y_pred), 1))], axis=1))[:, 2]
        
        # Calcular las zonas de Clarke
        _, zone = clarke_error_grid(pd.Series(y_test_desnormalized), pd.Series(y_pred_desnormalized), "Error de Clarke")
        zone_percentage_df = zone_percentages_new("Model", zone)
        print(f"Porcentajes de zonas: {zone_percentage_df}")

        # Extraer solo los valores numéricos y devolverlos como una lista
        zone_percentages = [
            zone_percentage_df['A_zone'],
            zone_percentage_df['B_zone'],
            zone_percentage_df['C_zone'],
            zone_percentage_df['D_zone'],
            zone_percentage_df['E_zone']
        ]

        return loss, len(self.x_test), {
            "rmse": rmse,
            "pearson": pearson_corr,
            "zone_A": zone_percentages[0],
            "zone_B": zone_percentages[1],
            "zone_C": zone_percentages[2],
            "zone_D": zone_percentages[3],
            "zone_E": zone_percentages[4],
        }
      


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    node_id = context.node_id 
     # `partition-id` uniquely identifies the client
    X_train, X_test, y_train, y_test, archivo, scaler = load_data_crosssilo(partition_id)
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
