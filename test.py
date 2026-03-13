"""
Evaluation script: compares AST adversaries against a Monte Carlo baseline
across two victim versions (V1 baseline, V2 hardened).

Metrics collected per (victim, method) cell:
  1. Failure rate (%)
  2. IS-estimated P(failure) under nominal N(0,1) — AST only
  3. Average trajectory NLL under N(0,1)
  4. Average time-to-failure (steps), conditioned on failure episodes
  5. Average minimum portfolio value ($)

Outputs:
  results/table.csv                   — full metrics table
  results/figure1_trajectories.png    — representative attack vs nominal paths
  results/figure2_failure_histogram.png — failure-time distributions
"""

import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from validation import TradingValidationEnv

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

N_EPISODES = 500  # episodes per (victim, method) cell
LOG_NORM_CONST = 0.5 * np.log(2 * np.pi)  # = 0.9189...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nominal_log_prob(epsilon: float) -> float:
    """Log probability of epsilon under N(0, 1)."""
    return -0.5 * epsilon ** 2 - LOG_NORM_CONST


def run_ast_episodes(model: PPO, env: TradingValidationEnv, n: int):
    """
    Run n stochastic AST episodes and collect all metrics including IS weights.

    Returns a list of episode dicts with keys:
      failed, ttf, min_portfolio, traj_nll, is_weight,
      prices, portfolio_values (for the first failure found)
    """
    results = []
    first_failure = None

    for _ in range(n):
        obs, _ = env.reset()
        done = False

        obs_traj, act_traj = [], []
        prices, portfolios = [], []
        episode_nll = 0.0
        ttf = None

        while not done:
            # Stochastic sampling (not deterministic) so IS weights are valid
            action, _ = model.predict(obs, deterministic=False)
            obs_traj.append(obs.copy())
            act_traj.append(action.copy())

            eps = float(action[0])
            episode_nll += -nominal_log_prob(eps)

            obs, _, done, _, _ = env.step(action)
            prices.append(float(obs[0]))
            portfolios.append(float(obs[1]))

            if obs[1] < 700.0 and ttf is None:
                ttf = len(prices)

        failed = ttf is not None

        # --- Importance weight ---
        # w = prod_t [ p_nominal(eps_t) / p_adversary(eps_t | s_t) ]
        obs_arr = np.array(obs_traj, dtype=np.float32)
        act_arr = np.array(act_traj, dtype=np.float32)
        obs_tensor = torch.FloatTensor(obs_arr).to(model.device)
        act_tensor = torch.FloatTensor(act_arr).to(model.device)

        with torch.no_grad():
            _, log_probs_adv, _ = model.policy.evaluate_actions(obs_tensor, act_tensor)

        log_probs_nom = torch.FloatTensor(
            [-0.5 * a[0] ** 2 - LOG_NORM_CONST for a in act_arr]
        ).to(model.device)

        log_weight = (log_probs_nom - log_probs_adv).sum().item()
        # Clamp to avoid overflow; -500 preserves tiny-but-nonzero weights
        log_weight = np.clip(log_weight, -500, 50)
        is_weight = np.exp(log_weight)

        ep = {
            "failed": failed,
            "ttf": ttf if failed else None,
            "min_portfolio": min(portfolios),
            "traj_nll": episode_nll / max(len(act_traj), 1),
            "is_weight": is_weight,
            "prices": prices,
            "portfolios": portfolios,
        }
        results.append(ep)

        if failed and first_failure is None:
            first_failure = ep

    return results, first_failure


def run_mc_episodes(env: TradingValidationEnv, n: int):
    """
    Run n Monte Carlo episodes using N(0,1) noise.

    Returns a list of episode dicts (same schema as AST, is_weight=1 always).
    """
    results = []
    first_failure = None

    for _ in range(n):
        obs, _ = env.reset()
        done = False

        prices, portfolios = [], []
        episode_nll = 0.0
        ttf = None

        while not done:
            action = np.random.normal(0, 1, size=(1,)).astype(np.float32)
            eps = float(action[0])
            episode_nll += -nominal_log_prob(eps)
            obs, _, done, _, _ = env.step(action)
            prices.append(float(obs[0]))
            portfolios.append(float(obs[1]))
            if obs[1] < 700.0 and ttf is None:
                ttf = len(prices)

        failed = ttf is not None
        ep = {
            "failed": failed,
            "ttf": ttf if failed else None,
            "min_portfolio": min(portfolios),
            "traj_nll": episode_nll / max(len(prices), 1),
            "is_weight": 1.0,
            "prices": prices,
            "portfolios": portfolios,
        }
        results.append(ep)
        if failed and first_failure is None:
            first_failure = ep

    return results, first_failure


def summarize(episodes, method: str):
    """Compute the 5 metrics from a list of episode dicts."""
    n = len(episodes)
    failures = [e for e in episodes if e["failed"]]

    failure_rate = len(failures) / n * 100.0

    # IS-estimated P(failure) — only meaningful for AST
    if method == "ast":
        is_p_failure = sum(e["is_weight"] for e in failures) / n
    else:
        is_p_failure = float("nan")

    avg_nll = np.mean([e["traj_nll"] for e in episodes])

    ttf_values = [e["ttf"] for e in failures]
    avg_ttf = np.mean(ttf_values) if ttf_values else float("nan")

    avg_min_portfolio = np.mean([e["min_portfolio"] for e in episodes])

    return {
        "failure_rate": failure_rate,
        "is_p_failure": is_p_failure,
        "avg_nll": avg_nll,
        "avg_ttf": avg_ttf,
        "avg_min_portfolio": avg_min_portfolio,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

configs = [
    {"victim": "v1", "adversary_path": "adversary_v1"},
    {"victim": "v2", "adversary_path": "adversary_v2"},
]

all_stats = {}
representative = {}  # store a representative failure path per (victim, method)

for cfg in configs:
    v = cfg["victim"]
    adv_path = cfg["adversary_path"]
    print(f"\n{'='*60}")
    print(f"Victim {v.upper()} | Loading adversary from {adv_path}")
    print(f"{'='*60}")

    env = TradingValidationEnv(victim_version=v)

    # Load model (try .zip extension first)
    for path in (adv_path + ".zip", adv_path):
        try:
            model = PPO.load(path, env=env)
            print(f"  Loaded model from {path}")
            break
        except Exception:
            continue

    # AST episodes
    print(f"  Running {N_EPISODES} AST episodes...")
    ast_eps, ast_first_fail = run_ast_episodes(model, env, N_EPISODES)
    ast_stats = summarize(ast_eps, method="ast")
    all_stats[(v, "ast")] = ast_stats

    # MC episodes
    print(f"  Running {N_EPISODES} Monte Carlo episodes...")
    mc_eps, mc_first_fail = run_mc_episodes(env, N_EPISODES)
    mc_stats = summarize(mc_eps, method="mc")
    all_stats[(v, "mc")] = mc_stats

    # Store all episode data for plotting
    nom_ep = mc_first_fail if mc_first_fail else mc_eps[0]
    representative[v] = {
        "ast_fail": ast_first_fail,
        "nom": nom_ep,
        "ast_all_eps": ast_eps,
        "mc_all_eps": mc_eps,
        "ast_all_ttf": [e["ttf"] for e in ast_eps if e["failed"]],
        "mc_all_ttf": [e["ttf"] for e in mc_eps if e["failed"]],
    }

    env.close()

# ---------------------------------------------------------------------------
# Print and save results table
# ---------------------------------------------------------------------------

metrics_order = [
    ("failure_rate",     "Failure Rate (%)"),
    ("is_p_failure",     "IS-estimated P(failure)"),
    ("avg_nll",          "Avg NLL (nats/step)"),
    ("avg_ttf",          "Avg Time-to-Failure (steps)"),
    ("avg_min_portfolio","Avg Min Portfolio ($)"),
]

print(f"\n{'='*60}")
print("FINAL RESULTS TABLE")
print(f"{'='*60}")
header = f"{'Metric':<32} {'V1 AST':>12} {'V1 MC':>12} {'V2 AST':>12} {'V2 MC':>12}"
print(header)
print("-" * len(header))

rows = []
for key, label in metrics_order:
    v1a = all_stats[("v1", "ast")][key]
    v1m = all_stats[("v1", "mc")][key]
    v2a = all_stats[("v2", "ast")][key]
    v2m = all_stats[("v2", "mc")][key]

    def fmt(x):
        if np.isnan(x):
            return "N/A"
        if key == "failure_rate":
            return f"{x:.1f}%"
        if key == "is_p_failure":
            if x == 0.0:
                return "0"
            return f"{x:.3e}"
        if key in ("avg_ttf", "avg_min_portfolio"):
            return f"{x:.1f}"
        return f"{x:.3f}"

    print(f"{label:<32} {fmt(v1a):>12} {fmt(v1m):>12} {fmt(v2a):>12} {fmt(v2m):>12}")
    rows.append([label, fmt(v1a), fmt(v1m), fmt(v2a), fmt(v2m)])

with open(os.path.join("results", "table.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "V1 AST", "V1 MC", "V2 AST", "V2 MC"])
    writer.writerows(rows)
print("\nSaved results/table.csv")

# ---------------------------------------------------------------------------
# Figure 1: All trajectories overlaid — market price and portfolio value
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "AST Adversarial Attack vs. Nominal (Monte Carlo) Trajectories\n"
    "(500 episodes each; failures bold, non-failures faded)",
    fontsize=12, fontweight="bold"
)

victim_labels = {"v1": "Victim V1 (Baseline, 5-step lag)", "v2": "Victim V2 (Hardened, 3-step lag)"}

for col, v in enumerate(["v1", "v2"]):
    rep = representative[v]
    ast_eps_all = rep["ast_all_eps"]
    mc_eps_all  = rep["mc_all_eps"]

    # ---- Price trajectories ----
    ax = axes[0][col]
    # MC: all episodes in muted blue at very low alpha
    for ep in mc_eps_all:
        color = "steelblue" if ep["failed"] else "lightsteelblue"
        lw    = 1.0         if ep["failed"] else 0.4
        alpha = 0.5         if ep["failed"] else 0.07
        ax.plot(ep["prices"], color=color, linewidth=lw, alpha=alpha)
    # AST: non-failures in light gray, failures in red
    for ep in ast_eps_all:
        if not ep["failed"]:
            ax.plot(ep["prices"], color="lightcoral", linewidth=0.4, alpha=0.05)
    for ep in ast_eps_all:
        if ep["failed"]:
            ax.plot(ep["prices"], color="crimson", linewidth=0.6, alpha=0.15)
    # Highlighted representatives
    if rep["ast_fail"]:
        ax.plot(rep["ast_fail"]["prices"], color="crimson", linewidth=2.0,
                label="AST failure (highlight)", zorder=5)
    ax.plot(rep["nom"]["prices"], color="steelblue", linestyle="--",
            linewidth=1.5, alpha=0.85, label="Nominal (MC)", zorder=4)
    ax.set_title(f"{victim_labels[v]}\nMarket Price", fontsize=10)
    ax.set_ylabel("Asset Price ($)", fontsize=9)
    ax.set_xlabel("Simulation Steps", fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ---- Portfolio value trajectories ----
    ax = axes[1][col]
    for ep in mc_eps_all:
        color = "steelblue" if ep["failed"] else "lightsteelblue"
        lw    = 1.0         if ep["failed"] else 0.4
        alpha = 0.5         if ep["failed"] else 0.07
        ax.plot(ep["portfolios"], color=color, linewidth=lw, alpha=alpha)
    for ep in ast_eps_all:
        if not ep["failed"]:
            ax.plot(ep["portfolios"], color="lightcoral", linewidth=0.4, alpha=0.05)
    for ep in ast_eps_all:
        if ep["failed"]:
            ax.plot(ep["portfolios"], color="crimson", linewidth=0.6, alpha=0.15)
    if rep["ast_fail"]:
        ax.plot(rep["ast_fail"]["portfolios"], color="crimson", linewidth=2.0,
                label="AST failure (highlight)", zorder=5)
    ax.plot(rep["nom"]["portfolios"], color="steelblue", linestyle="--",
            linewidth=1.5, alpha=0.85, label="Nominal (MC)", zorder=4)
    ax.axhline(y=700, color="black", linestyle=":", linewidth=2.0,
               label="Failure threshold (\$700)", zorder=6)
    ax.set_title("Portfolio Value", fontsize=10)
    ax.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax.set_xlabel("Simulation Steps", fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
fig_path = os.path.join("results", "figure1_trajectories.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fig_path}")

# ---------------------------------------------------------------------------
# Figure 2: Time-to-failure histograms (AST vs MC, V1 vs V2)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Distribution of Time-to-Failure: AST vs. Monte Carlo",
    fontsize=13, fontweight="bold"
)

bins = np.arange(0, 105, 5)

for col, v in enumerate(["v1", "v2"]):
    rep = representative[v]
    ax = axes[col]

    ast_ttf = rep["ast_all_ttf"]
    mc_ttf  = rep["mc_all_ttf"]

    if ast_ttf:
        ax.hist(ast_ttf, bins=bins, alpha=0.65, color="crimson",
                label=f"AST (n={len(ast_ttf)} failures)", density=True)
        mean_ast = np.mean(ast_ttf)
        ax.axvline(mean_ast, color="darkred", linestyle="--", linewidth=1.5,
                   label=f"AST mean = {mean_ast:.1f} steps")
    if mc_ttf:
        ax.hist(mc_ttf, bins=bins, alpha=0.65, color="steelblue",
                label=f"MC  (n={len(mc_ttf)} failures)", density=True)
        mean_mc = np.mean(mc_ttf)
        ax.axvline(mean_mc, color="navy", linestyle="--", linewidth=1.5,
                   label=f"MC mean = {mean_mc:.1f} steps")

    fr_ast = all_stats[(v, "ast")]["failure_rate"]
    fr_mc  = all_stats[(v, "mc")]["failure_rate"]
    ax.set_title(
        f"{victim_labels[v]}\nAST failure rate: {fr_ast:.1f}%  |  MC failure rate: {fr_mc:.1f}%",
        fontsize=10
    )
    ax.set_xlabel("Steps until failure", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join("results", "figure2_failure_histogram.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fig_path}")

print("\nEvaluation complete. All results written to results/")
