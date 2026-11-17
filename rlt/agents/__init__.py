"""Agent registry."""
from .base import Agent, EnvSpec  # noqa: F401
from .dqn import DQN, DoubleDQN
from .offpolicy import SAC, TD3
from .onpolicy import A2C, PPO
from .qlearning import TabularQ
from .recurrent import PPOLSTM, RRL

REGISTRY = {
    "qlearning": TabularQ,
    "dqn": DQN,
    "ddqn": DoubleDQN,
    "a2c": A2C,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "rrl": RRL,
    "ppo_lstm": PPOLSTM,
}
DISCRETE_ONLY = {"qlearning", "dqn", "ddqn"}
CONTINUOUS_ONLY = {"sac", "td3", "rrl"}


def make_agent(name: str, spec: EnvSpec, params: dict | None = None, seed: int = 0) -> Agent:
    return REGISTRY[name](spec, params, seed)
