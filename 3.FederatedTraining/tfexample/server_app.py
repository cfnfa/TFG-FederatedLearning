"""tfexample: A Flower / TensorFlow app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from tfexample.task import load_model
from tfexample.task import load_data

def gen_evaluate_fn(x_test,y_test,):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model()
        model.set_weights(parameters_ndarrays)
        loss, rmse = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"centralized_rmse": rmse}

    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    rmses = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"rmse": sum(rmses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(load_model().get_weights())
    X_train, X_test, y_train, y_test,archivo=load_data(11)

    # Define the strategy
    strategy =FedAvg(
        fraction_fit=context.run_config["fraction-fit"], # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=4,  # Never sample less than X clients for training
        min_available_clients=4, # Wait until all X clients are available
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=gen_evaluate_fn(X_train, y_train)
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
