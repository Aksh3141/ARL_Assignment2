
import os

def _pick_gl_backend():
    
    for backend in ("glfw", "egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:    
            import mujoco
            
            return backend
        except Exception:
            continue
    return None

_backend = _pick_gl_backend()
_RENDER_OK = _backend is not None
if _RENDER_OK:
    print(f"GL backend: {_backend.upper()}")
else:
    print("WARNING: No GL backend found. GIF recording will be skipped.")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy
import gymnasium as gym


CONFIG = {
    "envs":             ["Hopper-v4", "HalfCheetah-v4"],
    "seeds":            [0, 1, 2],
    "total_timesteps":  1_000_000,
    "steps_per_update": 8192,
    "gamma":            0.995,
    "lam":              0.97,
    "max_kl":           0.01,
    "cg_iters":         20,
    "cg_damping":       0.01,
    "backtrack_iters":  15,
    "backtrack_coeff":  0.8,
    "vf_lr":            1e-3,
    "vf_iters":         10,
    "vf_batch_size":    512,
    "hidden_size":      256,
    "n_layers":         3,
    "log_std_init":     -0.5,
    "norm_rewards":     True,
    "norm_obs":         True,
    "eval_episodes":    100,
    "gif_episodes":     3,
    "gif_fps":          30,
    "render_every":     2,
}


class RunningNormalizer:
    def __init__(self, shape, clip=10.0, epsilon=1e-8):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = epsilon
        self.clip    = clip
        self.epsilon = epsilon

    def update(self, x):
        x      = np.asarray(x, dtype=np.float64)
        batch  = x[None] if x.ndim == len(self.mean.shape) else x.reshape(-1, *self.mean.shape)
        n      = len(batch)
        mean_b = batch.mean(0)
        var_b  = batch.var(0)
        total  = self.count + n
        delta  = mean_b - self.mean
        self.mean  = self.mean + delta * n / total
        self.var   = (self.var * self.count + var_b * n +
                      delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(x, -self.clip, self.clip).astype(np.float32)


def mlp(in_dim, out_dim, hidden, n_layers, act=nn.Tanh):
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _ortho_init(net, gain_hidden=np.sqrt(2), gain_out=0.01):
    layers = list(net.children())
    linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
    for l in linear_layers[:-1]:
        nn.init.orthogonal_(l.weight, gain=gain_hidden)
        nn.init.zeros_(l.bias)
    nn.init.orthogonal_(linear_layers[-1].weight, gain=gain_out)
    nn.init.zeros_(linear_layers[-1].bias)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n_layers=3, log_std_init=-0.5):
        super().__init__()
        self.net     = mlp(obs_dim, act_dim, hidden, n_layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        _ortho_init(self.net, gain_out=0.01)

    def forward(self, obs):
        mean = self.net(obs)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs):
        dist = self.forward(obs)
        a    = dist.sample()
        return a, dist.log_prob(a).sum(-1)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden=256, n_layers=3):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden, n_layers)
        _ortho_init(self.net, gain_out=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx: idx + n].view_as(p))
        idx += n


def flat_grad(loss, params, retain=False, create=False):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain,
        create_graph=create, allow_unused=True
    )
    return torch.cat([
        g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
        for g, p in zip(grads, params)
    ])


def compute_gae(rewards, values, dones, gamma, lam):
    T, adv, last = len(rewards), torch.zeros(len(rewards)), 0.0
    for t in reversed(range(T)):
        mask   = 1.0 - dones[t]
        delta  = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last   = delta + gamma * lam * mask * last
        adv[t] = last
    return adv, adv + values[:-1]


def fisher_vector_product(policy, obs, v, damping):
    dist    = policy(obs)
    mean_kl = torch.mean(
        torch.distributions.kl_divergence(
            Normal(dist.mean.detach(), dist.stddev.detach()), dist
        ).sum(-1)
    )
    params = list(policy.parameters())
    grads  = flat_grad(mean_kl, params, retain=True, create=True)
    gv     = (grads * v.detach()).sum()
    fvp    = flat_grad(gv, params, retain=False)
    return fvp + damping * v


def conjugate_gradient(policy, obs, b, n_iters, damping):
    x, r, p = torch.zeros_like(b), b.clone(), b.clone()
    rr = torch.dot(r, r)
    for _ in range(n_iters):
        Fp     = fisher_vector_product(policy, obs, p, damping)
        alpha  = rr / (torch.dot(p, Fp) + 1e-8)
        x      = x + alpha * p
        r      = r - alpha * Fp
        rr_new = torch.dot(r, r)
        p      = r + (rr_new / (rr + 1e-8)) * p
        rr     = rr_new
        if rr < 1e-10:
            break
    return x


def surrogate_and_kl(policy, obs, acts, old_logp, advantages):
    dist  = policy(obs)
    logp  = dist.log_prob(acts).sum(-1)
    ratio = torch.exp(logp - old_logp)
    surr  = (ratio * advantages).mean()
    kl    = torch.distributions.kl_divergence(
                Normal(dist.mean.detach(), dist.stddev.detach()), dist
            ).sum(-1).mean()
    return surr, kl


def trpo_update(policy, vf, vf_optim, batch, cfg):
    obs        = batch["obs"]
    acts       = batch["acts"]
    old_logp   = batch["logp"].detach()
    advantages = batch["adv"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = batch["ret"]

    params  = list(policy.parameters())
    surr, _ = surrogate_and_kl(policy, obs, acts, old_logp, advantages)
    g       = flat_grad(surr, params, retain=True)

    step_dir  = conjugate_gradient(policy, obs, g, cfg["cg_iters"], cfg["cg_damping"])
    Fs        = fisher_vector_product(policy, obs, step_dir, cfg["cg_damping"])
    sFs       = torch.dot(step_dir, Fs)
    step_size = torch.sqrt(2.0 * cfg["max_kl"] / (sFs + 1e-8))
    full_step = step_size * step_dir

    old_params  = flat_params(policy).clone()
    old_surr, _ = surrogate_and_kl(policy, obs, acts, old_logp, advantages)
    success, new_surr, new_kl = False, old_surr, torch.tensor(0.0)

    for i in range(cfg["backtrack_iters"]):
        scale = cfg["backtrack_coeff"] ** i
        set_flat_params(policy, old_params + scale * full_step)
        new_surr, new_kl = surrogate_and_kl(policy, obs, acts, old_logp, advantages)
        if new_surr > old_surr and new_kl <= cfg["max_kl"]:
            success = True
            break

    if not success:
        set_flat_params(policy, old_params)

    n   = obs.shape[0]
    bs  = cfg["vf_batch_size"]
    for _ in range(cfg["vf_iters"]):
        idx = torch.randperm(n)
        for start in range(0, n, bs):
            mb   = idx[start: start + bs]
            loss = F.mse_loss(vf(obs[mb]), returns[mb])
            vf_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vf.parameters(), max_norm=0.5)
            vf_optim.step()

    return dict(surr=new_surr.item(), kl=new_kl.item(),
                vf_loss=loss.item(), success=success)



def collect_rollout(env, policy, vf, obs_norm, rew_norm, n_steps,
                    gamma, lam, device, cfg):
    obs_buf, act_buf, rew_buf = [], [], []
    done_buf, logp_buf, val_buf = [], [], []

    raw_obs, _ = env.reset()
    done, ep_ret, ep_rets = False, 0.0, []

    for _ in range(n_steps):
        if cfg["norm_obs"]:
            obs_norm.update(raw_obs)
            obs = obs_norm.normalize(raw_obs)
        else:
            obs = raw_obs.astype(np.float32)

        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            act, logp = policy.act(obs_t)
            val       = vf(obs_t)

        obs_buf.append(obs_t); act_buf.append(act)
        logp_buf.append(logp); val_buf.append(val)

        raw_obs, rew, terminated, truncated, _ = env.step(act.cpu().numpy())
        done    = terminated or truncated
        ep_ret += rew

        if cfg["norm_rewards"]:
            rew_norm.update(np.array([rew]))
            rew_scaled = rew / (np.sqrt(rew_norm.var[0]) + 1e-8)
        else:
            rew_scaled = float(rew)

        rew_buf.append(torch.tensor(rew_scaled, dtype=torch.float32))
        done_buf.append(torch.tensor(float(done), dtype=torch.float32))

        if done:
            ep_rets.append(ep_ret)
            ep_ret  = 0.0
            raw_obs, _ = env.reset()

    last_obs = obs_norm.normalize(raw_obs) if cfg["norm_obs"] else raw_obs.astype(np.float32)
    with torch.no_grad():
        last_val = vf(torch.FloatTensor(last_obs).to(device)) * (1.0 - float(done))

    vals_t   = torch.cat([torch.stack(val_buf), last_val.unsqueeze(0)])
    adv, ret = compute_gae(
        torch.stack(rew_buf), vals_t, torch.stack(done_buf), gamma, lam
    )
    return dict(obs=torch.stack(obs_buf), acts=torch.stack(act_buf),
                logp=torch.stack(logp_buf), adv=adv, ret=ret), ep_rets



def train_one_seed(env_name, seed, cfg, device):
    set_seed(seed)
    env     = gym.make(env_name)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy   = GaussianPolicy(obs_dim, act_dim, cfg["hidden_size"],
                               cfg["n_layers"], cfg["log_std_init"]).to(device)
    vf       = ValueFunction(obs_dim, cfg["hidden_size"], cfg["n_layers"]).to(device)
    vf_optim = torch.optim.Adam(vf.parameters(), lr=cfg["vf_lr"], eps=1e-5)
    obs_norm = RunningNormalizer(shape=(obs_dim,))
    rew_norm = RunningNormalizer(shape=(1,))

    log_ts, log_rets, t = [], [], 0
    print(f"  [{env_name} | seed={seed}] training ...")

    while t < cfg["total_timesteps"]:
        batch, ep_rets = collect_rollout(
            env, policy, vf, obs_norm, rew_norm,
            cfg["steps_per_update"], cfg["gamma"], cfg["lam"], device, cfg
        )
        info  = trpo_update(policy, vf, vf_optim, batch, cfg)
        t    += cfg["steps_per_update"]

        if ep_rets:
            mean_ret = np.mean(ep_rets)
            log_ts.append(t)
            log_rets.append(mean_ret)
            tick = "OK" if info["success"] else "--"
            print(f"    t={t:>8d}  ret={mean_ret:>8.1f}  "
                  f"kl={info['kl']:.4f}  [{tick}]")

    env.close()
    return log_ts, log_rets, policy, obs_norm


def evaluate(env_name, policy, obs_norm, n_episodes, device, norm_obs):
    env  = gym.make(env_name)
    rets = []
    for _ in range(n_episodes):
        raw_obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            obs = obs_norm.normalize(raw_obs) if norm_obs else raw_obs.astype(np.float32)
            with torch.no_grad():
                act = policy(torch.FloatTensor(obs).to(device)).mean
            raw_obs, rew, term, trunc, _ = env.step(act.cpu().numpy())
            done    = term or trunc
            ep_ret += rew
        rets.append(ep_ret)
    env.close()
    return np.mean(rets), np.std(rets)


def record_gif(env_name, policy, obs_norm, path, n_episodes, fps,
               render_every, device, norm_obs):
    if not _RENDER_OK:
        print(f"  GIF skipped for {env_name} (no GL backend).")
        return

    _vdisplay = None
    try:
        from xvfbwrapper import Xvfb
        _vdisplay = Xvfb(width=1280, height=720, colordepth=24)
        _vdisplay.start()
    except Exception:
        pass 

    try:
        env    = gym.make(env_name, render_mode="rgb_array")
        frames = []
        for _ in range(n_episodes):
            raw_obs, _ = env.reset()
            done, step = False, 0
            while not done:
                if step % render_every == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                obs = obs_norm.normalize(raw_obs) if norm_obs else raw_obs.astype(np.float32)
                with torch.no_grad():
                    act = policy(torch.FloatTensor(obs).to(device)).mean
                raw_obs, _, term, trunc, _ = env.step(act.cpu().numpy())
                done = term or trunc
                step += 1
        env.close()

        if frames:
            imageio.mimsave(path, frames, fps=fps)
            print(f"  GIF saved -> {path}  ({len(frames)} frames)")
        else:
            print(f"  WARNING: No frames captured for {env_name}.")

    except Exception as exc:
        print(f"  WARNING: GIF recording failed: {exc}")
        print("  Install fix:  pip install pyopengl glfw")
    finally:
        if _vdisplay is not None:
            try:
                _vdisplay.stop()
            except Exception:
                pass


def plot_curves(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, env_name in zip(axes, CONFIG["envs"]):
        seed_data = results[env_name]
        all_ts    = sorted({t for ts, _ in seed_data.values() for t in ts})
        arr       = np.array([
            np.interp(all_ts, ts, rets) for ts, rets in seed_data.values()
        ])
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(all_ts, mean, linewidth=2, label="Mean return")
        ax.fill_between(all_ts, mean - std, mean + std, alpha=0.3, label="+/- 1 std")
        ax.set_title(env_name, fontsize=14)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episodic Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("TRPO Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "trpo_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved -> {path}")



def main():
    out_dir   = "trpo_outputs"
    eval_path = os.path.join(out_dir, "eval_results.txt")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(eval_path):
        os.remove(eval_path)

    device = torch.device("cpu")
    print(f"Device: {device}\n")

    results   = {}
    best_data = {}

    for env_name in CONFIG["envs"]:
        print(f"\n{'='*60}\n  {env_name}\n{'='*60}")
        results[env_name] = {}
        best_final        = -np.inf

        for seed in CONFIG["seeds"]:
            ts, rets, policy, obs_norm = train_one_seed(
                env_name, seed, CONFIG, device
            )
            results[env_name][seed] = (ts, rets)
            if rets and rets[-1] > best_final:
                best_final          = rets[-1]
                best_data[env_name] = (deepcopy(policy), deepcopy(obs_norm))

        policy, obs_norm = best_data[env_name]

        print(f"\n  Evaluating {env_name} for {CONFIG['eval_episodes']} episodes ...")
        mean_ret, std_ret = evaluate(
            env_name, policy, obs_norm,
            CONFIG["eval_episodes"], device, CONFIG["norm_obs"]
        )
        print(f"  {env_name}:  mean={mean_ret:.2f}  std={std_ret:.2f}")
        with open(eval_path, "a") as f:
            f.write(f"{env_name}: mean={mean_ret:.2f}, std={std_ret:.2f} "
                    f"(over {CONFIG['eval_episodes']} episodes)\n")

        gif_path = os.path.join(out_dir, f"{env_name.replace('-', '_')}.gif")
        record_gif(
            env_name, policy, obs_norm, gif_path,
            CONFIG["gif_episodes"], CONFIG["gif_fps"],
            CONFIG["render_every"], device, CONFIG["norm_obs"]
        )

    plot_curves(results, out_dir)

    print(f"\nDeliverables in: ./{out_dir}/")
    print("  trpo_training_curves.png")
    print("  eval_results.txt")
    print("  Hopper_v4.gif")
    print("  HalfCheetah_v4.gif")


if __name__ == "__main__":
    main()