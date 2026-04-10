import argparse
import numpy as np
import os

from src.utils import (
    ALGORITHMS, get_model_path, is_websocket_env,
    check_env_exist, get_saved_hyperparams, restore_env,
    get_saved_stats
)
from src.route import get_routes
from src.websocket import find_free_port, run_socket, stop_socket
from src.plots import Plots


def evaluate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment ID")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(ALGORITHMS.keys()), help="RL algorithm")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of enjoying episodes")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: INFO)")
    parser.add_argument("--plot-results", action="store_true", help="Plot results")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--folder", type=str, default="trained-agents", help="Log folder")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID (default: 0: latest, -1: no exp folder)")
    parser.add_argument("--load-checkpoint", type=int,
                        help="Load checkpoint instead of last model if available" 
                             "you must pass the number of timesteps corresponding to it")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False,
                        help="Load last checkpoint instead of last model if available")
    parser.add_argument("--render", action="store_true", default=False, help="Render environment")
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
    episode_rewards, episode_lengths = [], []
    try:
        if is_websocket:
            ioloop, server = run_socket(port=find_free_port(), routes=get_routes(env=env))
            env.unwrapped.connection_event.wait()
        for episode in range(args.n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _states = model.predict(observation=obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if args.render and not is_websocket:
                    env.render()
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if args.verbose > 0:
                print(f'Total reward for episode {episode} is {episode_reward}')
                print(f'Total length of episode {episode} is {episode_length}')

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    if ioloop and server:
        stop_socket(ioloop, server)
    env.close()

    if args.plot_results and len(episode_rewards) > 0:
        plots = Plots()
        plots.add_plot(title="Episode Rewards", x_label="Episode", y_label="Reward", values=episode_rewards)
        plots.add_plot(title="Episode Lengths", x_label="Episode", y_label="Length", values=episode_lengths)
        plots.show()


if __name__ == "__main__":
    evaluate()
