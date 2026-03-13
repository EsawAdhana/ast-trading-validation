import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingValidationEnv(gym.Env):
    """
    Adversarial stress-testing environment for algorithmic trading strategies.

    The adversary controls per-step Gaussian noise (epsilon) injected into the
    market return. Two victim versions are supported:
      - "v1": SMA-50/200 crossover with a 5-step exit lag (baseline, fragile).
      - "v2": Same SMA-50/200 crossover but with a faster 3-step exit lag instead
              of 5. The faster exit reduces the window in which the adversary can
              keep the bot invested while driving down prices, increasing the
              adversarial sample complexity: more training timesteps are required
              to discover a reliable 30% drawdown exploit compared to V1.

    The reward signal for the adversary combines:
      - Portfolio damage (scaled)
      - A likelihood penalty (-0.05 * epsilon^2) that keeps actions close to
        the nominal N(0,1) distribution, which is required for valid IS weights.
      - A large crash bonus (+1000) when the 30% drawdown threshold is crossed.
    """

    def __init__(self, victim_version: str = "v1"):
        super(TradingValidationEnv, self).__init__()

        assert victim_version in ("v1", "v2"), "victim_version must be 'v1' or 'v2'"
        self.victim_version = victim_version

        # Adversary action: epsilon (noise) in [-3, 3]
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

        # Observation: [price, portfolio, SMA50, SMA200, is_invested]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.initial_portfolio = 1000.0

        # Exit lag: consecutive SMA50 < SMA200 steps required before selling
        self._exit_lag = 3 if victim_version == "v2" else 5

        # Placeholders — populated properly in reset()
        self.price = 100.0
        self.portfolio = self.initial_portfolio
        self.prices_history = [100.0] * 200
        self.step_count = 0
        self._sma_below_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Always start in a bull market so the victim is invested from step 0
        base_price = np.random.uniform(150, 200)
        self.prices_history = [base_price * (1 - 0.001 * i) for i in range(200)][::-1]
        self.price = base_price
        self.portfolio = self.initial_portfolio
        self.step_count = 0
        self._sma_below_steps = 0
        return self._get_obs(), {}

    def _compute_rsi(self, period: int = 14) -> float:
        """Wilder RSI over the last `period` price changes."""
        if len(self.prices_history) < period + 1:
            return 50.0  # neutral — not enough history
        prices = self.prices_history[-(period + 1):]
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _get_obs(self):
        sma50 = np.mean(self.prices_history[-50:])
        sma200 = np.mean(self.prices_history[-200:])
        is_invested = 1.0 if self._sma_below_steps < self._exit_lag else 0.0
        return np.array(
            [self.price, self.portfolio, sma50, sma200, is_invested], dtype=np.float32
        )

    def step(self, action):
        epsilon = float(action[0])
        prev_portfolio = self.portfolio

        # 1. Market dynamics: adversary injects noise into the return
        self.price = self.price * (1.0 + 0.0001 + 0.02 * epsilon)
        self.prices_history.append(self.price)

        # 2. Victim logic — SMA crossover with version-dependent exit lag
        sma50 = np.mean(self.prices_history[-50:])
        sma200 = np.mean(self.prices_history[-200:])
        if sma50 <= sma200:
            self._sma_below_steps = min(self._sma_below_steps + 1, self._exit_lag)
        else:
            self._sma_below_steps = 0

        is_invested = 1.0 if self._sma_below_steps < self._exit_lag else 0.0

        # 3. Update portfolio
        if is_invested:
            return_pct = (self.price - self.prices_history[-2]) / self.prices_history[-2]
            self.portfolio *= (1.0 + return_pct)

        # 4. Build observation after portfolio update
        obs = self._get_obs()

        # 5. Adversary reward
        damage_reward = (prev_portfolio - self.portfolio) / 2.0
        # Likelihood penalty: keeps epsilon near N(0,1), required for valid IS
        likelihood_penalty = -0.05 * (epsilon ** 2)
        reward = damage_reward + likelihood_penalty

        # 6. Failure check (30% drawdown from initial)
        terminated = False
        if self.portfolio < 700.0:
            terminated = True
            reward += 1000.0

        self.step_count += 1
        if self.step_count >= 100:
            terminated = True

        return obs, reward, terminated, False, {}


# --- Quick sanity check / Monte Carlo baseline ---
if __name__ == "__main__":
    for version in ("v1", "v2"):
        print(f"\nRunning Monte Carlo Baseline (Random Noise) — Victim {version.upper()}...")
        env = TradingValidationEnv(victim_version=version)
        failures = 0
        runs = 1000

        for _ in range(runs):
            obs, _ = env.reset()
            done = False
            while not done:
                random_noise = np.random.normal(0, 1, size=(1,))
                obs, reward, done, _, _ = env.step(random_noise)
                if obs[1] < 700:
                    failures += 1
                    break

        print(f"  Result: {failures} failures out of {runs} runs ({failures/runs*100:.2f}%)")
