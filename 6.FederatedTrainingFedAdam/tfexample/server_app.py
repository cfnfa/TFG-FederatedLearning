"""Author: Clara Fuertes Novillo"""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from tfexample.task import load_model
from tfexample.task import load_data_crossdevice
import numpy as np
from flwr.server.strategy import FedAdam

def gen_evaluate_fn(x_test,y_test,):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model()
        model.set_weights(parameters_ndarrays)
        loss, rmse = model.evaluate(x_test, y_test, verbose=2)
        return loss, {"centralized_rmse": rmse}

    return evaluate

def weighted_average(metrics):
    """Calcular la media ponderada de las métricas de los clientes, incluyendo las zonas de Clarke."""
    
    # Calcular las métricas ponderadas 
    weighted_rmse = sum(num_examples * m["rmse"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)
    weighted_pearson = sum(num_examples * m["pearson"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)

    # Zonas de Clarke
    
    weighted_zone_A = sum(num_examples * m["zone_A"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)
    weighted_zone_B = sum(num_examples * m["zone_B"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)
    weighted_zone_C = sum(num_examples * m["zone_C"]for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)
    weighted_zone_D = sum(num_examples * m["zone_D"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)
    weighted_zone_E = sum(num_examples * m["zone_E"] for num_examples, m in metrics) / sum(num_examples for num_examples, _ in metrics)

    return {"federated_evaluate_rmse": weighted_rmse,"federated_evaluate_pearson": weighted_pearson,"federated_clarke_zone_A": weighted_zone_A,"federated_clarke_zone_B": weighted_zone_B, "federated_clarke_zone_C": weighted_zone_C,"federated_clarke_zone_D": weighted_zone_D,"federated_clarke_zone_E": weighted_zone_E,}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(load_model().get_weights())
    X_train, X_test, y_train, y_test,archivo,scaler=load_data_crossdevice(11)

    # Define the FEDADAM PERSONALIZED strategy
    strategy =FedAdam(
        fraction_evaluate=1,  # 0,8 para cross-silo, 1 para cross-device
        min_fit_clients=8,  # 8 para cross-device
        min_available_clients=8, # Wait until X clients are available
        initial_parameters=parameters,
        evaluate_fn=gen_evaluate_fn(X_train, y_train),
        evaluate_metrics_aggregation_fn=weighted_average,
        eta= 0.1,
        eta_l=0.1,
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
