import argparse
import os

from src.utils import ALGORITHMS, ConvertToDict, check_env_exist
from src.manager import TrainingManager


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Gym or Websocket Environment")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(ALGORITHMS.keys()), help="RL algorithm")
    parser.add_argument("--n-timesteps", type=int, default=-1, help="Number of training steps")
    parser.add_argument("--hyperparams", type=str, nargs="+", action=ConvertToDict,
                        help="Change default hyperparameters")
    parser.add_argument("--tensorboard-log", type=str, default="logs", help="Tensorboard log dir")
    parser.add_argument("--trained-agent", type=str, default="", help="Path to pretrained agent")
    parser.add_argument("--save-freq", type=int, default=0, help="Frequency to save model")
    parser.add_argument("--log-folder", type=str, default="trained-agents", help="Log folder")
    parser.add_argument("--verbose", type=int, default=0, help="Enable preview")
    parser.add_argument("--conf-file", type=str, default=None, help="Custom yaml file")
    args = parser.parse_args()
    env_name = args.env

    check_env_exist(env_name)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    training_manager = TrainingManager(
        args=args,
        env_name=args.env,
        algo=args.algo,
        tensorboard_log=args.tensorboard_log,
        trained_agent=args.trained_agent,
        n_timesteps=args.n_timesteps,
        save_freq=args.save_freq,
        log_folder=args.log_folder,
        verbose=args.verbose,
        hyperparameters=args.hyperparams,
        config=args.conf_file
    )

    model = training_manager.setup_model()
    if model is not None:
        training_manager.learn_model(model)
        training_manager.save_model(model)


if __name__ == "__main__":
    train()
