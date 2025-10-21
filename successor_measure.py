import numpy as np
import random


def compute_occ_matrices_from_policy(env, horizon, pi):
    dim = env.num_states * env.num_actions
    occupancy_matrices = [np.zeros((dim, dim)) for _ in range(horizon)]

    for s in env.state_to_idx:
        print(s)
        for a in env.actions:
            start_idx = env.sa_index(s, a)
            # Initialize distribution: mass 1 at next_state from (s,a)
            current_distribution = np.zeros((env.num_states,))
            if env.is_valid_transition(s, a)[0]:
                _, next_state = env.step(s,a)
                if hasattr(pi, 'last_action'):
                    # import pdb; pdb.set_trace()
                    pi.last_action = a
                current_distribution[env.state_to_idx[next_state]] = 1.0
            else:
                continue

            for t in range(horizon):
                # Spread current_distribution over actions using pi(a|s)
                for state_idx, prob in enumerate(current_distribution):
                    if prob == 0:
                        continue
                    state = env.idx_to_state[state_idx]
                    for a_prime in env.actions:
                        if not env.is_valid_transition(state, a_prime)[0]:
                            continue
                        prob_a_prime = pi.pi_probability(state, a_prime)
                        if prob_a_prime == 0:
                            continue
                        next_idx = env.sa_index(state, a_prime)
                        occupancy_matrices[t][start_idx, next_idx] += prob * prob_a_prime

                # Prepare next distribution for the next timestep
                next_distribution = np.zeros_like(current_distribution)
                for state_idx, prob in enumerate(current_distribution):
                    # import pdb; pdb.set_trace()
                    if prob == 0:
                        continue
                    state = env.idx_to_state[state_idx]
                    for a_prime in env.actions:
                        if not env.is_valid_transition(state, a_prime)[0]:
                            continue
                        _, next_state = env.step(state, a_prime)
                        if hasattr(pi, 'last_action'):
                            pi.last_action = a_prime
                        prob_a_prime = pi.pi_probability(state, a_prime)
                        next_distribution[env.state_to_idx[next_state]] += prob * prob_a_prime

                current_distribution = next_distribution

    return occupancy_matrices



def learn_SM_td(env, dataset, gamma=0.95, num_samples=int(1e6), alpha=0.1, k_shift=0):
    dim = env.num_states * env.num_actions
    M_flat = np.zeros((dim, dim))

    # Preprocess all valid (s,a) → (s',a') transitions
    transitions = []
    for traj in dataset:
        for t in range(len(traj) - 1 - k_shift):
            s, a = traj[t]
            s_next, a_next = traj[t + 1 + k_shift]

            # if not env.is_valid_transition(s, a)[0]:
            #     continue
            # if not env.is_valid_transition(s_next, a_next)[0]:
            #     continue

            transitions.append((s, a, s_next, a_next))

    # Perform TD-style updates: M(sa) ← γ M(s'a') + one-step transition
    for _ in range(num_samples):
        s, a, s_next, a_next = random.choice(transitions)

        sa_idx = env.sa_index(s, a)
        next_sa_idx = env.sa_index(s_next, a_next)

        # M_flat[sa_idx] += gamma * M_flat[next_sa_idx] + np.eye(dim)[next_sa_idx]
        M_flat[sa_idx] = (1 - alpha) * M_flat[sa_idx] + alpha * (np.eye(dim)[next_sa_idx] + gamma * M_flat[next_sa_idx])

    return M_flat*(1-gamma)







def learn_SM_model_based(env, dataset, gamma=0.95, num_iters=100):
    dim = env.num_states * env.num_actions
    transition_counts = np.zeros((dim, dim))

    for trajectory in dataset:
        for t in range(len(trajectory) - 1):
            (s, a) = trajectory[t]
            (s_prime, a_prime) = trajectory[t + 1]

            if not env.is_valid_transition(s, a)[0]:
                continue
            if not env.is_valid_transition(s_prime, a_prime)[0]:
                continue

            sa_idx = env.sa_index(s, a)
            sa_prime_idx = env.sa_index(s_prime, a_prime)
            transition_counts[sa_idx, sa_prime_idx] += 1

    # Normalize to get P_pi
    row_sums = transition_counts.sum(axis=1, keepdims=True) + 1e-8
    P_pi = transition_counts / row_sums

    # Power series to compute successor measure
    M = P_pi.copy()
    P_power = np.eye(dim)

    for _ in range(num_iters):
        P_power = gamma * P_power @ P_pi
        M += P_power

    return (1 - gamma) * M, P_pi


