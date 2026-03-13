"""
Train two PPO adversaries — one per victim version.

  adversary_v1.zip  — trained against the baseline SMA crossover (V1, 5-step exit lag)
  adversary_v2.zip  — trained against the hardened V2 victim (3-step exit lag)

V2 requires more timesteps because the shorter exit lag narrows the inertia window
the adversary can exploit, increasing adversarial sample complexity.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from validation import TradingValidationEnv


def train_adversary(victim_version: str, total_timesteps: int, save_path: str):
    env = make_vec_env(
        lambda: TradingValidationEnv(victim_version=victim_version), n_envs=4
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
    )
    print(f"\n{'='*60}")
    print(f"Training adversary against Victim {victim_version.upper()}")
    print(f"Timesteps: {total_timesteps:,}  |  Save path: {save_path}")
    print(f"{'='*60}")
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved adversary to {save_path}.zip")
    env.close()


if __name__ == "__main__":
    # Adversary A1: baseline victim (easier — 200k steps sufficient)
    train_adversary(
        victim_version="v1",
        total_timesteps=200_000,
        save_path="adversary_v1",
    )

    # Adversary A2: hardened victim (harder — 400k steps for better convergence)
    train_adversary(
        victim_version="v2",
        total_timesteps=400_000,
        save_path="adversary_v2",
    )

    print("\nAll adversaries trained and saved.")
