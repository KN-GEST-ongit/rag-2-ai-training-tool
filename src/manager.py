import os
import gymnasium as gym
import yaml
import argparse
import pickle

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from src.utils import (
    ALGORITHMS, get_latest_run_id, is_websocket_env,
    get_saved_hyperparams, restore_env, BLOCKED_ALGORITHMS_WS,
    get_saved_stats
)
from src.route import get_routes
from src.websocket import find_free_port, run_socket, stop_socket
from src.wrappers import NormObsWrapper, NormRewWrapper
from collections import OrderedDict
from typing import Any


class TrainingManager:
    def __init__(
            self,
            args: argparse.Namespace,
            env_name: str,
            algo: str,
            tensorboard_log: str,
            trained_agent: str,
            n_timesteps: int,
            save_freq: int,
            log_folder: str,
            verbose: int,
            hyperparameters: dict | None,
            config: str | None
    ):
        super().__init__()
        self.args = args
        self.env_name = env_name
        self.algo = algo
        self.tensorboard_log = os.path.join(tensorboard_log, self.env_name)
        self.trained_agent = trained_agent
        self.n_timesteps = n_timesteps
        self.save_freq = save_freq
        self.log_folder = log_folder
        self.verbose = verbose
        self.custom_hyperparameters = hyperparameters or {}
        self.config = config or f"hyperparams/{self.algo}.yml"

        # File paths
        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = os.path.join(
            self.log_path, f"{self.env_name}_{get_latest_run_id(self.log_path, self.env_name) + 1}"
        )
        self.params_path = f"{self.save_path}/{self.env_name}"
        self.stats_path = f"{os.path.dirname(self.trained_agent)}/{self.env_name}" if self.trained_agent != "" else ""

        # Other
        self.env = None
        self.is_websocket = False
        self.continue_training = self.trained_agent.endswith(".zip") or os.path.isfile(self.trained_agent)
        self.state_stack = 1
        self.norm_obs = False
        self.norm_reward = False

    def _load_model(self, env: gym.Env, hyperparams: dict[str, Any]) -> BaseAlgorithm:
        hyperparams.pop("policy", None)
        hyperparams.pop("policy_kwargs", None)

        model = ALGORITHMS[self.algo].load(
            path=self.trained_agent,
            env=env,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams
        )
        return model

    def _get_default_parameters(self) -> dict[str, Any]:
        self.env = gym.make(self.env_name)
        self.is_websocket = is_websocket_env(self.env)
        if self.is_websocket and self.algo in BLOCKED_ALGORITHMS_WS:
            raise ValueError(f"Env {self.env_name} doesn't support {self.algo} yet.")
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            if self.algo == "ARS":
                raise AssertionError("ARS algorithm doesn't support Dict observation space")
            policy = "MultiInputPolicy"
        elif isinstance(obs_space, gym.spaces.Box):
            if len(obs_space.shape) == 3:
                if self.algo == "ARS":
                    raise AssertionError("ARS algorithm doesn't support RGB observation space")
                policy = "CnnPolicy"
            else:
                policy = "MlpPolicy" if self.algo != "ARS" else "LinearPolicy"
        else:
            policy = "MlpPolicy" if self.algo != "ARS" else "LinearPolicy"
        return {"policy": policy, "n_timesteps": 1e5}

    def _read_parameters(self) -> dict[str, Any]:
        if self.stats_path == "":
            with open(self.config) as f:
                hyperparams_dict = yaml.safe_load(f)
            if hyperparams_dict is not None and self.env_name in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_name]
            else:
                hyperparams = self._get_default_parameters()
        else:
            hyperparams = get_saved_hyperparams(stats_path=self.stats_path)
        if self.n_timesteps > 0:
            hyperparams.update({"n_timesteps": self.n_timesteps})
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])
        # ---
        if self.custom_hyperparameters is not None:
            hyperparams.update(self.custom_hyperparameters)
        hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
        return hyperparams

    def _preprocess_hyperparams(self, hyperparams: dict[str, Any]) -> dict[str, Any]:
        _hyperparams = hyperparams.copy()
        if "state_stack" in _hyperparams.keys():
            self.state_stack = _hyperparams["state_stack"]
            del _hyperparams["state_stack"]
        if "norm_obs" in _hyperparams.keys():
            self.norm_obs = _hyperparams["norm_obs"]
            del _hyperparams["norm_obs"]
        if "norm_reward" in _hyperparams.keys():
            self.norm_reward = _hyperparams["norm_reward"]
            del _hyperparams["norm_reward"]
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in _hyperparams.keys() and isinstance(_hyperparams[kwargs_key], str):
                _hyperparams[kwargs_key] = eval(_hyperparams[kwargs_key])
        del _hyperparams["n_timesteps"]
        return _hyperparams

    def _setup_env(self, config: dict[str, Any]) -> gym.Env:
        if self.continue_training:
            return restore_env(self.env_name, config=config, stats=get_saved_stats(stats_path=self.stats_path))
        return gym.make(self.env_name)

    def setup_model(self) -> BaseAlgorithm:
        os.makedirs(self.params_path, exist_ok=True)
        save_hyperparams = self._read_parameters()
        _hyperparams = self._preprocess_hyperparams(save_hyperparams)
        if self.env is None:
            self.env = self._setup_env(save_hyperparams)
        self.is_websocket = is_websocket_env(self.env)
        if self.is_websocket and self.algo in BLOCKED_ALGORITHMS_WS:
            raise ValueError(f"Env {self.env_name} doesn't support {self.algo} yet.")
        if self.continue_training:
            model = self._load_model(env=self.env, hyperparams=_hyperparams)
        else:
            if self.norm_obs:
                self.env = NormObsWrapper(self.env)
            if self.norm_reward:
                self.env = NormRewWrapper(self.env)
            if self.state_stack > 1:
                self.env = gym.wrappers.FrameStackObservation(self.env, self.state_stack)
            model = ALGORITHMS[self.algo](
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                **_hyperparams
            )
        self._save_parameters(save_hyperparams)
        return model

    def learn_model(self, model: BaseAlgorithm) -> None:
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.save_path,
            name_prefix=self.env_name,
        ) if self.save_freq > 0 else None
        ioloop = server = None
        try:
            if self.is_websocket:
                ioloop, server = run_socket(port=find_free_port(), routes=get_routes(env=self.env))
                self.env.unwrapped.connection_event.wait()
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=checkpoint_callback,
            )
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        if ioloop and server:
            stop_socket(ioloop, server)

    def _save_parameters(self, hyperparams: dict[str, Any]) -> None:
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(hyperparams, f)

        with open(os.path.join(self.params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

    def _save_normalize(self):
        stats = {}
        if self.norm_obs:
            stats["obs_mean"] = self.env.obs_rms.mean
            stats["obs_var"] = self.env.obs_rms.var
        if self.norm_reward:
            stats["rew_mean"] = self.env.return_rms.mean
            stats["rew_var"] = self.env.return_rms.var
        with open(os.path.join(self.params_path, "normalize.pkl"), "wb") as f:
            pickle.dump(stats, f)

    def save_model(self, model: BaseAlgorithm) -> None:
        model.save(f"{self.save_path}/{self.env_name}")
        if self.norm_obs or self.norm_reward:
            self._save_normalize()
