import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
import os

from TreasureHunt.env import TreasureHunt

# ================= CONFIG =================
GAMMA = 0.99
ITERATIONS = 500
MC_SAMPLES_V = 4000
ROLLOUT_MAX_STEPS = 200

ALPHA_VALUES = [0.01, 0.02, 0.05, 0.1]

# ================= ENV =================
locations = {
    'ship': [(0,0)],
    'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),
             (0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
    'fort': [(9,9)],
    'pirate': [(4,7),(8,5)],
    'treasure': [(4,0),(1,9)]
}

# ================= UTIL =================
def preprocess(state, num_states):
    one_hot = np.zeros(num_states, dtype=np.float32)
    one_hot[state] = 1.0
    return one_hot

# ================= POLICY =================
class CPIPolicy:
    def __init__(self, num_states, alpha):
        self.greedy_policies = []
        self.num_states = num_states
        self.alpha = alpha

    def sample_action(self, s_idx):
        for g in reversed(self.greedy_policies):
            if np.random.rand() < self.alpha:
                return np.argmax(g[s_idx])
        return np.random.randint(4)

    def add_greedy(self, g):
        self.greedy_policies.append(g)

# ================= SAFE STEP =================
def safe_step(env, action):
    out = env.step(action)
    if len(out) == 3:
        state, reward, done = out
    else:
        state, reward = out
        done = False
    return state, reward, done

# ================= MONTE CARLO VALUE =================
def mc_value_estimate(env, policy, gamma):
    num_states = env.num_states

    # random restart
    s = np.random.randint(num_states)
    env.reset(state=s)

    total_return = 0
    discount = 1.0

    state = s

    for _ in range(ROLLOUT_MAX_STEPS):
        a = policy.sample_action(state)
        state, r, done = safe_step(env, a)

        total_return += discount * r
        discount *= gamma

        if done:
            break

    return s, total_return

# ================= DATA =================
def collect_value_data(env, policy, gamma, n_samples):
    X, Y = [], []

    for _ in tqdm(range(n_samples), desc="Collecting Value Data"):
        s, G = mc_value_estimate(env, policy, gamma)
        X.append(preprocess(s, env.num_states))
        Y.append(G)

    return np.array(X), np.array(Y)

# ================= GREEDY =================
def compute_greedy_policy(env, model):
    policy = np.zeros((env.num_states, 4), dtype=int)

    for s in range(env.num_states):
        best_a = 0
        best_val = -1e9

        V_s = model.predict(preprocess(s, env.num_states).reshape(1,-1))[0]

        for a in range(4):
            env.reset(state=s)
            s_next, r, done = safe_step(env, a)

            V_next = model.predict(preprocess(s_next, env.num_states).reshape(1,-1))[0]

            adv = r + GAMMA * V_next - V_s

            if adv > best_val:
                best_val = adv
                best_a = a

        policy[s, best_a] = 1

    return policy

# ================= EVAL =================
def evaluate_policy(env, policy):
    state, _ = env.reset()
    total = 0

    for _ in range(200):
        a = policy.sample_action(state)
        state, r, done = safe_step(env, a)
        total += r
        if done:
            break

    return total

# ================= HEATMAP =================
def plot_value_heatmap(env, model, save_path):
    num_treasures = env.num_treasures                 
    num_combos    = 2 ** num_treasures                
    grid_size     = env.n                              

    # Predict V for all states at once
    all_features = np.eye(env.num_states, dtype=np.float32)  
    all_values   = model.predict(all_features)                

    fig, axes = plt.subplots(1, num_combos, figsize=(5 * num_combos, 5))
    fig.suptitle("Value Function Heatmaps per Treasure Combination", fontsize=14)

    for combo_idx in range(num_combos):
        # States for this treasure combination
        start = combo_idx * grid_size * grid_size      
        end   = start + grid_size * grid_size         

        # Reshape 100 values -> 10x10 grid
        grid_values = all_values[start:end].reshape(grid_size, grid_size)


        indicator = env.treasure_from_index[combo_idx]
        label = f"Treasure state: {indicator}"

        ax = axes[combo_idx]
        im = ax.imshow(grid_values, cmap='viridis')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Y (col)")
        ax.set_ylabel("X (row)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {save_path}")

# ================= TRAIN =================
def train_cpi(alpha):
    env = TreasureHunt(locations=locations)
    policy = CPIPolicy(env.num_states, alpha=alpha)

    model = MLPRegressor(
        hidden_layer_sizes=(256,),
        activation='relu',
        solver='adam',
        max_iter=500,
        warm_start=True
    )

    rewards_log = []

    for it in range(ITERATIONS):
        print(f"\n===== Iteration {it} =====")

        # ---- collect MC data ----
        X, Y = collect_value_data(env, policy, GAMMA, MC_SAMPLES_V)

        # ---- train V ----
        model.fit(X, Y)

        # ---- improve policy ----
        greedy = compute_greedy_policy(env, model)
        policy.add_greedy(greedy)

        # ---- evaluate ----
        reward = evaluate_policy(env, policy)
        rewards_log.append(reward)
        print("Reward:", reward)

    return env, policy, model, rewards_log

# ================= RUN =================
if __name__ == "__main__":

    for alpha in ALPHA_VALUES:
        print(f"\n{'='*50}")
        print(f"Training with alpha = {alpha}")
        print(f"{'='*50}")

        folder = f"alpha_{alpha}"
        os.makedirs(folder, exist_ok=True)

        env, policy, model, rewards_log = train_cpi(alpha)

        # ---- reward plot ----
        plt.figure()
        plt.plot(rewards_log)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Undiscounted Reward")
        plt.title(f"CPI Returns (alpha={alpha})")
        plt.savefig(os.path.join(folder, "rewards.png"))
        plt.close()

        # ---- heatmap (4 subplots, one per treasure combo) ----
        plot_value_heatmap(env, model, os.path.join(folder, "value_heatmap.png"))

        # ---- policy vis ----
        final_policy = policy.greedy_policies[-1]
        env.visualize_policy(final_policy, path=os.path.join(folder, "policy_vis.png"))
        env.visualize_policy_execution(final_policy, path=os.path.join(folder, "policy.gif"))

        print(f"Saved all outputs to folder: {folder}")