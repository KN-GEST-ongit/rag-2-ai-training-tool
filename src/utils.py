import os
import glob
import numpy as np
import gymnasium as gym
import difflib
import yaml
import pickle


from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import ARS, TRPO, QRDQN, TQC, MaskablePPO
from typing import Any
from argparse import Action
from src.wrappers import NormObsWrapper, NormRewWrapper


def is_websocket_env(env: gym.Env) -> bool:
    from src.envs.web_env import WebsocketEnv
    while isinstance(env, gym.Wrapper):
        env = env.env
    return isinstance(env, WebsocketEnv)


def check_env_exist(env_name: str) -> None:
    registered_envs = set(gym.registry.keys())
    if env_name not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_name, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_name} not found in gym registry, you maybe meant {closest_match}?")


# Copied from source
ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "A2C": A2C,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,

    # sb3_contrib
    "ARS": ARS,
    "QRDQN": QRDQN,
    "TQC": TQC,
    "TRPO": TRPO,
    "MaskablePPO": MaskablePPO
}

BLOCKED_ALGORITHMS_WS: set[str] = {"DDPG", "SAC", "TD3", "ARS", "TQC", "MaskablePPO"}


def get_latest_run_id(log_path: str, env_name: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


def get_model_path(
    exp_id: int,
    folder: str,
    algo: str,
    env_name: str,
    load_best: bool = False,
    load_checkpoint: str | None = None,
    load_last_checkpoint: bool = False,
) -> tuple[str, str, str]:
    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_name)
        print(f"Loading latest experiment, id={exp_id}")

    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_name}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    model_name = f"{algo}-{env_name}"

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        name_prefix = f"best-model-{model_name}"
    elif load_checkpoint is not None:
        model_path = os.path.join(log_path, f"{env_name}_{load_checkpoint}_steps.zip")
        name_prefix = f"checkpoint-{load_checkpoint}-{model_name}"
    elif load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, f"{env_name}_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_name}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        name_prefix = f"checkpoint-{step_count(model_path)}-{model_name}"
    else:
        model_path = os.path.join(log_path, f"{env_name}.zip")
        name_prefix = f"final-model-{model_name}"

    found = os.path.isfile(model_path)
    if not found:
        raise ValueError(f"No model found for {algo} on {env_name}, path: {model_path}")

    return name_prefix, model_path, log_path


class ConvertToDict(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key, value = arguments.split(":", 1)
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def get_saved_hyperparams(stats_path: str) -> dict[str, Any]:
    hyperparams: dict[str, Any] = {}
    if not os.path.isdir(stats_path):
        return hyperparams
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            with open(config_file) as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
    return hyperparams


def get_saved_stats(stats_path: str) -> dict[str, Any]:
    stats: dict = {}
    if not os.path.isdir(stats_path):
        return stats
    else:
        normalize_file = os.path.join(stats_path, "normalize.pkl")
        if os.path.isfile(normalize_file):
            with open(normalize_file, "rb") as f:
                stats = pickle.load(f)
    return stats


def restore_env(env_name: str, config: dict[str, Any], stats: dict, training=True):
    env = gym.make(env_name)
    n_stack = config.get("state_stack", 0)
    norm_obs = config.get("norm_obs", False)
    norm_reward = config.get("norm_reward", False)
    if norm_obs:
        env = NormObsWrapper(env, stats, training)
    if norm_reward:
        env = NormRewWrapper(env, stats, training)
    if n_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, n_stack)
    return env
