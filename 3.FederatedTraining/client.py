import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from model import load_lstm_model
from typing import Dict, Tuple
import tensorflow as tf
from dataset import cargar_datos
from flwr.common import Context, ParametersRecord
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


#el cliente es una instancia de la clase de cliente de flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.x_train, self.y_train, self.x_test, self.y_test = data
        

        # a model that is randomly initialised at first
        self.model= load_lstm_model()
        #self.device=algo para gpu en tensorflow
    

    def fit(self, parameters, config):
        # Convertir parámetros a pesos
        weights = parameters_to_ndarrays(parameters)
        print(f"Pesos recibidos para el cliente (fit): {len(weights)} capas.")
        self.model.set_weights(weights)
        # Train using tf.data.Dataset
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)

        return (
            self.model.get_weights(), 
            len (self.x_train), 
            {}
        )

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return float(loss), len(self.x_test), {"accuracy": accuracy}


def generate_client_fn():
    def client_fn(cid: str):
        # Convert FlowerClient to a proper Flower client using to_client()
        data= cargar_datos(int(cid))
        return FlowerClient(data).to_client()
    return client_fn

