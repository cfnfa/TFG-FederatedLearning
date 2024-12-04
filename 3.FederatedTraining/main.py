import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn

from server import get_evaluate_fn, get_on_fit_config
from model import load_lstm_model
from flwr.common import ndarrays_to_parameters

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    save_path = HydraConfig.get().runtime.output_dir

    
    
    sequence_length = 12  # Same as in your dataset

    #Creamos los clientes
    client_fn = generate_client_fn()
    model = load_lstm_model()
    initial_parameters = ndarrays_to_parameters(model.get_weights())
    ## 4. Define your strategy
    # Using FedAvg strategy for Federated Learning with LSTM
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Fraction of clients to use for training in each round
        min_fit_clients=cfg.num_clients_per_round_fit,  # Number of clients to sample for fit()
        fraction_evaluate=0.1,  # Fraction of clients to evaluate the model in each round
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # Number of clients to sample for evaluation
        min_available_clients=cfg.num_clients,  # Total clients in the simulation
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),  # Custom config for client-side training
        evaluate_fn=get_evaluate_fn(cfg.sequence_length, input_dim=cfg.input_dim),  # Custom evaluate function for the LSTM model
    )

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    # Setting GPU resources dynamically to use the two GPUs:
    num_gpus_per_client = 0.5  # Each client gets 50% of one GPU
    num_clients_to_use = cfg.num_clients
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=num_clients_to_use,  # total number of clients (1 for centralized or as per config for FL)
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds  # Configuring the number of rounds in FL
        ),
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 2,  # Assuming each client can use 2 CPUs for parallelism
            "num_gpus": num_gpus_per_client,  # Each client uses half a GPU
        },
    )

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.

    ## 6. Save your results
    ## 6. Save your results
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.

    results_path = Path(save_path) / "results.pkl"

    # Add the history returned by the strategy into a standard Python dictionary
    # You can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    results = {
        "history": history,
        "config": OmegaConf.to_yaml(cfg),  # Optionally save the full configuration for reproducibility
        "experiment_type": "centralized" if cfg.centralized_learning else "federated"
    }

    # Save the results as a Python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()