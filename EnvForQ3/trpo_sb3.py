import os

def _pick_gl_backend():
    for backend in ("glfw", "egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco  # noqa
            return backend
        except Exception:
            continue
    return None

_backend   = _pick_gl_backend()
_RENDER_OK = _backend is not None
print(f"GL backend: {_backend.upper() if _RENDER_OK else 'NONE (GIFs skipped)'}")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym

from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

BASE = {
    "total_timesteps": 1_000_000,
    "hidden_size":     256,
    "n_layers":        2,
    "log_std_init":    -0.5,
    "eval_episodes":   100,
    "gif_episodes":    3,
    "gif_fps":         30,
    "render_every":    2,
}

ENV_CONFIGS = {
    
    "Hopper-v4": {
        **BASE,
        "n_steps":                      8196,
        "batch_size":                   256,
        "gamma":                        0.997,
        "gae_lambda":                   0.98,
        "max_kl":                       0.015,
        "cg_max_steps":                 20,
        "cg_damping":                   0.01,
        "line_search_shrinking_factor": 0.8,
        "line_search_max_iter":         15,
        "n_critic_updates":             25,
        "learning_rate":                3e-4,
    },

    "HalfCheetah-v4": {
        **BASE,
        "n_steps":                      8192,
        "batch_size":                   512,
        "gamma":                        0.99,
        "gae_lambda":                   0.95,
        "max_kl":                       0.02,
        "cg_max_steps":                 15,
        "cg_damping":                   0.05,
        "line_search_shrinking_factor": 0.8,
        "line_search_max_iter":         10,
        "n_critic_updates":             10,
        "learning_rate":                1e-3,
    },
}

SEEDS = [0, 1, 2]
ENVS  = ["Hopper-v4", "HalfCheetah-v4"]


# =============================================================================
# CALLBACK
# =============================================================================
class ReturnLogger(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.log_ts   = []
        self.log_rets = []
        self._buf     = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._buf.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        if self._buf:
            self.log_ts.append(self.num_timesteps)
            self.log_rets.append(float(np.mean(self._buf)))
            print(f"    t={self.num_timesteps:>9d}  ret={self.log_rets[-1]:>8.1f}")
            self._buf = []


# =============================================================================
# TRAINING
# =============================================================================
def train_one_seed(env_name, seed, cfg):
    env = make_vec_env(env_name, n_envs=1, seed=seed)

    policy_kwargs = dict(
        net_arch      = [cfg["hidden_size"]] * cfg["n_layers"],
        activation_fn = nn.Tanh,
        log_std_init  = cfg["log_std_init"],
        ortho_init    = True,
    )

    model = TRPO(
        policy                       = "MlpPolicy",
        env                          = env,
        learning_rate                = cfg["learning_rate"],
        n_steps                      = cfg["n_steps"],
        batch_size                   = cfg["batch_size"],
        gamma                        = cfg["gamma"],
        cg_max_steps                 = cfg["cg_max_steps"],
        cg_damping                   = cfg["cg_damping"],
        line_search_shrinking_factor = cfg["line_search_shrinking_factor"],
        line_search_max_iter         = cfg["line_search_max_iter"],
        n_critic_updates             = cfg["n_critic_updates"],
        gae_lambda                   = cfg["gae_lambda"],
        target_kl                    = cfg["max_kl"],
        normalize_advantage          = True,
        seed                         = seed,
        device                       = "cpu",
        policy_kwargs                = policy_kwargs,
        verbose                      = 0,
    )

    cb = ReturnLogger()
    print(f"  [SB3 | {env_name} | seed={seed}] training ...")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=cb,
                progress_bar=False)
    env.close()
    return cb.log_ts, cb.log_rets, model


# =============================================================================
# EVALUATION — clean raw env, model.predict handles normalisation internally
# =============================================================================
def evaluate(env_name, model, n_episodes):
    env  = gym.make(env_name)
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(action)
            done    = term or trunc
            ep_ret += rew
        rets.append(ep_ret)
    env.close()
    return np.mean(rets), np.std(rets)


# =============================================================================
# GIF
# =============================================================================
def record_gif(env_name, model, path, cfg):
    if not _RENDER_OK:
        print(f"  GIF skipped for {env_name} (no GL backend).")
        return
    try:
        env    = gym.make(env_name, render_mode="rgb_array")
        frames = []
        for _ in range(cfg["gif_episodes"]):
            obs, _ = env.reset()
            done, step = False, 0
            while not done:
                if step % cfg["render_every"] == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
                step += 1
        env.close()
        if frames:
            imageio.mimsave(path, frames, fps=cfg["gif_fps"])
            print(f"  GIF -> {path}  ({len(frames)} frames)")
        else:
            print(f"  WARNING: no frames for {env_name}.")
    except Exception as e:
        print(f"  WARNING: GIF failed for {env_name}: {e}")


# =============================================================================
# PLOTTING
# =============================================================================
def plot_curves(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, env_name in zip(axes, ENVS):
        sd     = results[env_name]
        all_ts = sorted({t for ts, _ in sd.values() for t in ts})
        if not all_ts:
            continue
        arr       = np.array([np.interp(all_ts, ts, rets) for ts, rets in sd.values()])
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(all_ts, mean, linewidth=2, color="#ff7f0e", label="SB3 TRPO (mean)")
        ax.fill_between(all_ts, mean - std, mean + std,
                        alpha=0.25, color="#ff7f0e", label="+/- 1 std")
        ax.set_title(env_name, fontsize=14)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episodic Return")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SB3 TRPO Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "sb3_trpo_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves -> {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    out_dir   = "sb3_trpo_outputs"
    eval_path = os.path.join(out_dir, "eval_results.txt")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(eval_path):
        os.remove(eval_path)

    results     = {}
    best_models = {}

    for env_name in ENVS:
        cfg = ENV_CONFIGS[env_name]
        print(f"\n{'='*60}\n  {env_name}\n{'='*60}")
        results[env_name] = {}
        best_final = -np.inf

        for seed in SEEDS:
            ts, rets, model = train_one_seed(env_name, seed, cfg)
            results[env_name][seed] = (ts, rets)
            if rets and rets[-1] > best_final:
                best_final           = rets[-1]
                best_models[env_name] = model

        # evaluation
        print(f"\n  Evaluating {env_name} ({cfg['eval_episodes']} episodes)...")
        mean_ret, std_ret = evaluate(env_name, best_models[env_name], cfg["eval_episodes"])
        print(f"  {env_name}: mean={mean_ret:.2f}  std={std_ret:.2f}")
        with open(eval_path, "a") as f:
            f.write(f"{env_name}: mean={mean_ret:.2f}, std={std_ret:.2f} "
                    f"(over {cfg['eval_episodes']} episodes)\n")

        # GIF
        gif_path = os.path.join(out_dir, f"{env_name.replace('-','_')}_sb3.gif")
        record_gif(env_name, best_models[env_name], gif_path, cfg)

    plot_curves(results, out_dir)

    print(f"\nDeliverables in: ./{out_dir}/")
    print("  sb3_trpo_training_curves.png")
    print("  eval_results.txt")
    print("  Hopper_v4_sb3.gif")
    print("  HalfCheetah_v4_sb3.gif")


if __name__ == "__main__":
    main()