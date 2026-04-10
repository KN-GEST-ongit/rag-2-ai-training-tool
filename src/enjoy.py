import argparse
import os

from src.utils import (
    ALGORITHMS, get_model_path, is_websocket_env,
    check_env_exist, get_saved_hyperparams, restore_env,
    get_saved_stats
)
from src.route import get_routes
from src.websocket import find_free_port, run_socket, stop_socket


def enjoy() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment ID")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(ALGORITHMS.keys()), help="RL algorithm")
    parser.add_argument("--folder", type=str, default="trained-agents", help="Log folder")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID (default: 0: latest, -1: no exp folder)")
    parser.add_argument("--load-checkpoint", type=int,
                        help="Load checkpoint instead of last model if available" 
                             "you must pass the number of timesteps corresponding to it")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False,
                        help="Load last checkpoint instead of last model if available")
    args = parser.parse_args()

    env_name = args.env
    algo = args.algo
    folder = args.folder

    check_env_exist(env_name)

    _, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        False,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    stats_path = os.path.join(log_path, env_name)
    hyperparams = get_saved_hyperparams(stats_path)
    stats = get_saved_stats(stats_path)

    env = restore_env(
        env_name=env_name,
        config=hyperparams,
        stats=stats,
        training=False
    )
    is_websocket = is_websocket_env(env)

    model = ALGORITHMS[algo].load(path=model_path)

    ioloop = server = None
    try:
        if is_websocket:
            ioloop, server = run_socket(port=find_free_port(), routes=get_routes(env=env))
            env.unwrapped.connection_event.wait()
        while True:
            obs, info = env.reset()
            done = False

            while not done:
                action, _states = model.predict(observation=obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if not is_websocket:
                    env.render()
    except KeyboardInterrupt:
        print("Execution stopped by user.")

    if ioloop and server:
        stop_socket(ioloop, server)
    env.close()


if __name__ == "__main__":
    enjoy()
