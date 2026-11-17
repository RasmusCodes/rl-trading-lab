"""Do the agent implementations learn on problems with a known answer?

CartPole-v1 (discrete, max return 500) for DQN / Double DQN / A2C / PPO / PPO-LSTM;
Pendulum-v1 (continuous, ~-150 is solved, random ~-1200) for SAC / TD3 / PPO / A2C.
Each job runs in its own process (1 torch thread). Results -> runs/sanity/results.json
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOBS = [
    ("CartPole-v1", "dqn", 50_000, {"gamma": 0.99, "learning_starts": 1_000, "eps_frac": 0.16, "eps_end": 0.04,
                                    "grad_steps": 4, "hidden": (256, 256), "lr": 1e-3, "batch": 64, "tau": 0.01}),
    ("CartPole-v1", "ddqn", 50_000, {"gamma": 0.99, "learning_starts": 1_000, "eps_frac": 0.16, "eps_end": 0.04,
                                     "grad_steps": 4, "hidden": (256, 256), "lr": 1e-3, "batch": 64, "tau": 0.01}),
    ("CartPole-v1", "a2c", 150_000, {"n_steps": 8, "ent_coef": 0.0}),
    ("CartPole-v1", "ppo", 100_000, {"n_steps": 256, "minibatch": 256, "ent_coef": 0.0, "lr": 3e-4}),
    ("CartPole-v1", "ppo_lstm", 100_000, {"n_steps": 128, "chunk": 32}),
    ("Pendulum-v1", "sac", 20_000, {"learning_starts": 1_000, "grad_steps": 1, "hidden": (256, 256), "lr": 1e-3}),
    ("Pendulum-v1", "td3", 20_000, {"learning_starts": 1_000, "grad_steps": 1, "hidden": (256, 256),
                                    "explore_noise": 0.1}),
    ("Pendulum-v1", "ppo", 300_000, {"n_steps": 512, "minibatch": 256, "gamma": 0.9, "lam": 0.95, "lr": 1e-3,
                                     "epochs": 10, "log_std_init": 0.0, "target_kl": 0.0}),
    ("Pendulum-v1", "a2c", 300_000, {"n_steps": 16, "gamma": 0.9, "lr": 7e-4, "log_std_init": 0.0}),
]


def evaluate(agent, env_id, episodes=10, seed=123):
    import gymnasium as gym
    import numpy as np

    env = gym.make(env_id)
    disc = isinstance(env.action_space, gym.spaces.Discrete)
    rets = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        agent.reset(1)
        done, R = False, 0.0
        while not done:
            a = agent.act(np.asarray(o, np.float32)[None], deterministic=True)[0]
            a = int(a) if disc else np.clip(a, -1, 1) * env.action_space.high
            o, r, te, tr, _ = env.step(a)
            R += r
            done = te or tr
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def run(job):
    import torch

    torch.set_num_threads(1)
    from rlt.agents import EnvSpec, make_agent
    from rlt.agents.gym_adapter import GymVecAdapter

    env_id, algo, steps, params = job
    n_envs = 1 if algo in ("sac", "td3") else 8
    env = GymVecAdapter(env_id, n_envs, seed=0)
    spec = EnvSpec.from_env(env)
    agent = make_agent(algo, spec, {**params, "eval_every": 10**9}, seed=0)
    curve = []
    t0 = time.time()
    agent.learn(env, steps, log=lambda row: curve.append({k: row.get(k) for k in ("step", "ep_log_return")}))
    mean, sd = evaluate(agent, env_id)
    return {"env": env_id, "algo": algo, "steps": steps, "eval_return": mean, "eval_sd": sd,
            "seconds": round(time.time() - t0, 1), "curve": curve[-20:]}


if __name__ == "__main__":
    out = ROOT / "runs" / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(9, len(JOBS))) as ex:
        res = list(ex.map(run, JOBS))
    for r in res:
        print(f"{r['env']:12s} {r['algo']:9s} steps={r['steps']:7d} eval={r['eval_return']:8.1f} "
              f"(sd {r['eval_sd']:.1f}) {r['seconds']}s")
    (out / "results.json").write_text(json.dumps(res, indent=1))
