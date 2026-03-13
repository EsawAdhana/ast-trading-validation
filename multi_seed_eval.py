"""
Multi-seed evaluation: computes mean ± std for all Table 1 metrics
across N_TRIALS independent evaluation runs on the pre-trained models.

Each trial draws a fresh batch of N_EPISODES episodes with a different
numpy/gym random seed, giving trial-to-trial variance that reflects
the sampling uncertainty of each metric.

Outputs:
  results/table_with_std.csv   — full metrics with mean ± std
  results/table_stats.npy      — raw per-trial numbers for inspection
"""

import os
import csv
import warnings
import numpy as np
import torch
from stable_baselines3 import PPO
from validation import TradingValidationEnv

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

N_TRIALS   = 5      # independent evaluation batches
N_EPISODES = 500    # episodes per trial (same as original)
LOG_NORM_CONST = 0.5 * np.log(2 * np.pi)


def nominal_log_prob(epsilon: float) -> float:
    return -0.5 * epsilon ** 2 - LOG_NORM_CONST


def run_ast_episodes(model: PPO, env: TradingValidationEnv, n: int, seed: int):
    np.random.seed(seed)
    results = []
    for _ in range(n):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        done = False
        obs_traj, act_traj, portfolios = [], [], []
        episode_nll = 0.0
        ttf = None

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs_traj.append(obs.copy())
            act_traj.append(action.copy())
            eps = float(action[0])
            episode_nll += -nominal_log_prob(eps)
            obs, _, done, _, _ = env.step(action)
            portfolios.append(float(obs[1]))
            if obs[1] < 700.0 and ttf is None:
                ttf = len(portfolios)

        failed = ttf is not None

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
        log_weight = np.clip(log_weight, -500, 50)
        is_weight = np.exp(log_weight)

        results.append({
            "failed": failed,
            "ttf": ttf,
            "min_portfolio": min(portfolios),
            "traj_nll": episode_nll / max(len(act_traj), 1),
            "is_weight": is_weight,
        })
    return results


def run_mc_episodes(env: TradingValidationEnv, n: int, seed: int):
    np.random.seed(seed)
    results = []
    for _ in range(n):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        done = False
        portfolios = []
        episode_nll = 0.0
        ttf = None

        while not done:
            action = np.random.normal(0, 1, size=(1,)).astype(np.float32)
            eps = float(action[0])
            episode_nll += -nominal_log_prob(eps)
            obs, _, done, _, _ = env.step(action)
            portfolios.append(float(obs[1]))
            if obs[1] < 700.0 and ttf is None:
                ttf = len(portfolios)

        failed = ttf is not None
        results.append({
            "failed": failed,
            "ttf": ttf,
            "min_portfolio": min(portfolios),
            "traj_nll": episode_nll / max(len(portfolios), 1),
            "is_weight": 1.0,
        })
    return results


def summarize(episodes, method: str):
    n = len(episodes)
    failures = [e for e in episodes if e["failed"]]
    failure_rate = len(failures) / n * 100.0
    if method == "ast" and failures:
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


configs = [
    {"victim": "v1", "adversary_path": "adversary_v1"},
    {"victim": "v2", "adversary_path": "adversary_v2"},
]

# Collect per-trial statistics: trial_stats[(victim, method)][metric] = [val_t0, val_t1, ...]
trial_stats = {(v["victim"], m): {k: [] for k in ["failure_rate","is_p_failure","avg_nll","avg_ttf","avg_min_portfolio"]}
               for v in configs for m in ("ast", "mc")}

for cfg in configs:
    v = cfg["victim"]
    adv_path = cfg["adversary_path"]
    env = TradingValidationEnv(victim_version=v)

    for path in (adv_path + ".zip", adv_path):
        try:
            model = PPO.load(path, env=env)
            break
        except Exception:
            continue

    for trial in range(N_TRIALS):
        seed = 42 + trial * 100
        print(f"  Victim {v.upper()} | Trial {trial+1}/{N_TRIALS} | seed={seed}")

        ast_eps = run_ast_episodes(model, env, N_EPISODES, seed=seed)
        ast_s   = summarize(ast_eps, method="ast")
        for k, val in ast_s.items():
            trial_stats[(v, "ast")][k].append(val)

        mc_eps = run_mc_episodes(env, N_EPISODES, seed=seed + 1)
        mc_s   = summarize(mc_eps, method="mc")
        for k, val in mc_s.items():
            trial_stats[(v, "mc")][k].append(val)

    env.close()

# Compute mean ± std
metrics_order = [
    ("failure_rate",      "Failure Rate (%)"),
    ("is_p_failure",      "IS-estimated P(failure)"),
    ("avg_nll",           "Avg NLL (nats/step)"),
    ("avg_ttf",           "Avg Time-to-Failure (steps)"),
    ("avg_min_portfolio", "Avg Min Portfolio ($)"),
]

print("\n" + "="*80)
print("MULTI-SEED RESULTS (mean ± std over {} trials, {} eps/trial)".format(N_TRIALS, N_EPISODES))
print("="*80)

rows = []
for key, label in metrics_order:
    row = [label]
    for (v, m) in [("v1","ast"), ("v1","mc"), ("v2","ast"), ("v2","mc")]:
        vals = [x for x in trial_stats[(v, m)][key] if not np.isnan(x)]
        if not vals:
            mean_s, std_s = "N/A", ""
        else:
            mean_val = np.mean(vals)
            std_val  = np.std(vals)
            if key == "is_p_failure":
                mean_s = f"{mean_val:.3e}"
                std_s  = f"{std_val:.3e}"
            elif key == "failure_rate":
                mean_s = f"{mean_val:.1f}"
                std_s  = f"{std_val:.1f}"
            else:
                mean_s = f"{mean_val:.3f}"
                std_s  = f"{std_val:.3f}"
        cell = f"{mean_s} ± {std_s}" if std_s else mean_s
        print(f"  {label:<32} [{v.upper()} {m.upper()}]: {cell}")
        row.append(cell)
    rows.append(row)

with open(os.path.join("results", "table_with_std.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "V1 AST", "V1 MC", "V2 AST", "V2 MC"])
    writer.writerows(rows)

print("\nSaved results/table_with_std.csv")

# Also print LaTeX-friendly table rows
print("\n--- LaTeX table rows ---")
for key, label in metrics_order:
    cells = []
    for (v, m) in [("v1","ast"), ("v1","mc"), ("v2","ast"), ("v2","mc")]:
        vals = [x for x in trial_stats[(v, m)][key] if not np.isnan(x)]
        if not vals:
            cells.append("N/A")
        else:
            mean_val = np.mean(vals)
            std_val  = np.std(vals)
            if key == "is_p_failure":
                cells.append(f"${mean_val:.2e} \\pm {std_val:.1e}$")
            elif key == "failure_rate":
                cells.append(f"${mean_val:.1f} \\pm {std_val:.1f}$")
            else:
                cells.append(f"${mean_val:.3f} \\pm {std_val:.3f}$")
    print(f"  {label} & {' & '.join(cells)} \\\\")
