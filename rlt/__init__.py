"""rlt — a reinforcement-learning laboratory for trading research.

Layout
------
rlt.data        market data (download/cache, synthetic markets), causal features, regimes, walk-forward splits
rlt.env         vectorised trading environment: execution, costs, portfolio accounting, rewards
rlt.agents      Q-learning, DQN/Double DQN, A2C, PPO, SAC, TD3, recurrent RL (Moody-Saffell RRL, LSTM-PPO)
rlt.baselines   buy & hold, random, momentum, mean reversion, supervised models, portfolio optimisation
rlt.evaluation  metrics (Sharpe, PSR/DSR, drawdown, turnover, capacity), backtests, robustness
rlt.training    experiment runner, walk-forward orchestration, hyperparameter search
rlt.dashboard   local training dashboard (reads runs/)
"""

__version__ = "0.2.0"

# Bump when a change alters simulated P&L, features or folds. Recorded in every run's config/eval so analyses
# can refuse to mix versions.
#   1  initial environment (2026-11-13)
#   2  audit fixes (2026-11-14): limited liability + liquidation, evaluation never truncated by risk stops,
#      short-rebate haircut and 2008 short-sale ban, split-consistent dollar volume (ADV/impact/features),
#      ZN roll repair from IEF, eq10/pool19 training from 2001
ENV_VERSION = 2
