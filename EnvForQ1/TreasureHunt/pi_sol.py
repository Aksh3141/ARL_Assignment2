import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from general_env import TreasureHunt


def policy_evaluation(env, policy, gamma=0.95):
    S = env.num_states

    P_pi = np.zeros((S, S))
    for s in range(S):
        P_pi[s] = env.T[s, policy[s]]

    r = env.reward
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r)

    return V


def policy_improvement(env, V, gamma=0.95):
    S, A = env.num_states, env.num_actions
    new_policy = np.zeros(S, dtype=int)

    for s in range(S):
        Q_vals = []
        for a in range(A):
            Q = env.reward[s] + gamma * np.dot(env.T[s, a], V)
            Q_vals.append(Q)
        new_policy[s] = np.argmax(Q_vals)

    return new_policy


def policy_iteration(env, gamma=0.95, max_iters=50):
    S = env.num_states

    policy = np.random.randint(env.num_actions, size=S)
    V_norms = []

    for i in tqdm(range(max_iters)):
        V = policy_evaluation(env, policy, gamma)
        V_norms.append(np.linalg.norm(V))

        new_policy = policy_improvement(env, V, gamma)

        if np.all(policy == new_policy):
            print(f"Converged at iteration {i}")
            break

        policy = new_policy

    return policy, V, np.array(V_norms)




def evaluate_policy(env, policy, episodes=100):
    rewards = []
    treasures = []

    for _ in range(episodes):
        s = env.reset()
        total_reward = 0
        collected = 0

        done = False
        while not done:
            a = policy[s]
            s, r, done, info = env.step(a)

            total_reward += r
            if info.get("treasure_obtained", False):
                collected += 1

        rewards.append(total_reward)
        treasures.append(collected)

    return (
        np.mean(rewards), np.std(rewards),
        np.mean(treasures), np.std(treasures)
    )



class PolicyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, s):
        return self.policy[s]




def generate_gif(env, policy):
    wrapped = PolicyWrapper(policy)
    env.visualize_policy_execution(wrapped, "pi.gif")


def visualize_policy_map(env, policy):
    wrapped = PolicyWrapper(policy)
    env.visualize_policy(wrapped, path="pi_policy.png")




def run_all():
    locations = {
        'ship': [(0, 0)],
        'land': [(2,0),(2,1),(3,1),(0,5),(0,6),(1,5)],
        'fort': [(6,6)],
        'pirate': [(3,4),(5,3)],
        'treasure': [(3,0),(1,6)]
    }

    env = TreasureHunt(locations)

    # -------------------------
    # RUN PI
    # -------------------------
    policy, V, V_norms = policy_iteration(env)

    # -------------------------
    # TRAINING PLOT
    # -------------------------
    plt.plot(V_norms)
    plt.xlabel("Iterations")
    plt.ylabel("||V||_2")
    plt.title("Policy Iteration Convergence")
    plt.savefig("pi_training_plot.png")
    plt.show()

    # -------------------------
    # EVALUATION
    # -------------------------
    mean_r, std_r, mean_t, std_t = evaluate_policy(env, policy)

    

    # -------------------------
    # GIF
    # -------------------------
    generate_gif(env, policy)

    # -------------------------
    # POLICY MAP VISUALIZATION
    # -------------------------
    visualize_policy_map(env, policy)

    print("\n===== FINAL RESULTS (PI) =====")
    print(f"Cumulative Reward: {mean_r:.2f} ± {std_r:.2f}")
    print(f"Treasures Collected: {mean_t:.2f} ± {std_t:.2f}")


if __name__ == "__main__":
    run_all()