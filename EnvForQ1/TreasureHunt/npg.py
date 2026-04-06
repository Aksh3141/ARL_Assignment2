import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from general_env import TreasureHunt 


# Policy Class (Softmax)
class TabularPolicy:
    def __init__(self, num_states, num_actions, seed=0):
        np.random.seed(seed)
        self.theta = np.random.randn(num_states, num_actions) * 0.01
        self.num_actions = num_actions

    def get_probs(self):
        logits = self.theta - self.theta.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def get_action(self, s):
        probs = self.get_probs()[s]
        return np.random.choice(len(probs), p=probs)

    def get_greedy_policy(self):
        return np.argmax(self.get_probs(), axis=1)


# Exact Value Computation
def compute_V_Q_A(env, policy, gamma):
    pi = policy.get_probs()
    S, A = env.num_states, env.num_actions

    # P_pi
    P_pi = np.zeros((S, S))
    for s in range(S):
        for a in range(A):
            P_pi[s] += pi[s, a] * env.T[s, a]

    r_pi = env.reward.copy()

    # Solve (I - gamma P)V = r
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)

    # Q
    Q = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            Q[s, a] = env.reward[s] + gamma * np.dot(env.T[s, a], V)

    # Advantage
    A_adv = Q - V[:, None]

    return V, Q, A_adv

# NPG Update
def npg_update(policy, A_adv, lr, gamma):
    policy.theta += lr * (1/(1-gamma)) * A_adv

# REINFORCE Update
def reinforce_update(policy, A_adv, lr):
    pi = policy.get_probs()
    policy.theta += lr * (pi * A_adv)


# Training Loop
def train(env, algo="npg", seed=0, iters=1000, lr=0.01, gamma=0.95):
    policy = TabularPolicy(env.num_states, env.num_actions, seed)

    V_norms = []

    for _ in tqdm(range(iters)):
        V, Q, A_adv = compute_V_Q_A(env, policy, gamma)

        V_norms.append(np.linalg.norm(V))

        if algo == "npg":
            npg_update(policy, A_adv, lr, gamma)
        else:
            reinforce_update(policy, A_adv, lr)

    return policy, np.array(V_norms)


# Evaluation
def evaluate(env, policy, episodes=100):
    greedy = policy.get_greedy_policy()

    rewards = []
    treasures = []

    for _ in range(episodes):
        s = env.reset()
        total_reward = 0
        collected = 0

        done = False
        while not done:
            a = greedy[s]
            s, r, done, info = env.step(a)
            total_reward += r
            if info.get("treasure_obtained", False):
                collected += 1

        rewards.append(total_reward)
        treasures.append(collected)

    return np.mean(rewards), np.std(rewards), np.mean(treasures), np.std(treasures)


# Main Experiment
def run_experiment():
    locations = {
        'ship': [(0, 0)],
        'land': [(2,0),(2,1),(3,1),(0,5),(0,6),(1,5)],
        'fort': [(6,6)],
        'pirate': [(3,4),(5,3)],
        'treasure': [(3,0),(1,6)]
    }

    seeds = [0,1,2]
    iters = 1000

    npg_runs = []
    rein_runs = []

    for seed in seeds:
        env = TreasureHunt(locations)

        _, v_npg = train(env, "npg", seed, iters)
        _, v_rein = train(env, "reinforce", seed, iters)

        npg_runs.append(v_npg)
        rein_runs.append(v_rein)

    npg_runs = np.array(npg_runs)
    rein_runs = np.array(rein_runs)


    # Plot
    
    def plot_with_std(data, label):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        plt.plot(mean, label=label)
        plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)

    plot_with_std(npg_runs, "NPG")
    plot_with_std(rein_runs, "REINFORCE")

    plt.xlabel("Iterations")
    plt.ylabel("||V||_2")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig("training_plot.png")
    #plt.show()

    
    # Final Evaluation
    env = TreasureHunt(locations)

    npg_policy, _ = train(env, "npg", 0, iters)
    rein_policy, _ = train(env, "reinforce", 0, iters)

    npg_res = evaluate(env, npg_policy)
    rein_res = evaluate(env, rein_policy)

    

    # GIFs
    env.visualize_policy_execution(npg_policy, "npg.gif")
    env.visualize_policy_execution(rein_policy, "reinforce.gif")

    print("\nEvaluation Results:")
    print("NPG: Reward {:.2f}±{:.2f}, Treasure {:.2f}±{:.2f}".format(*npg_res))
    print("REINFORCE: Reward {:.2f}±{:.2f}, Treasure {:.2f}±{:.2f}".format(*rein_res))


if __name__ == "__main__":
    run_experiment()