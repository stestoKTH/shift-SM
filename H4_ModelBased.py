import numpy as np
from maze_environments import GridEnvironment
from policies import UnifRandomPolicy, GreedySMPolicy
import utils 
from successor_measure import learn_SM_model_based, learn_SM_td
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt


env = GridEnvironment('Medium-maze',horizontal_exp = 2, vertical_exp = 2)
unif_pi = UnifRandomPolicy(env)
num_trajs = np.round(np.logspace(0.0, 4.0, num=10)).astype(int)  
N_s = len(num_trajs)
rank = 40
dim = env.num_states*env.num_actions
N_pairs = 100
gamma = 0.95
horizon = 100
k_arr = [0, 2, 4, 8, 16]

dist_ks_learn_td = np.zeros([5,len(k_arr),N_s,N_pairs])
for i_d in range(5):
    for data_cnt in range(N_s):
        print(data_cnt)
        dataset = env.generate_dataset(unif_pi, num_trajectories = num_trajs[data_cnt])
        for k_cnt in range(len(k_arr)):
            print(k_cnt)
            M = learn_SM_td(env, dataset, gamma=gamma, num_samples=int(1e5), alpha=0.1, k_shift=k_arr[k_cnt])
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            Mr = utils.rank_r_approximation(U, S, Vt, r = rank)
            for cnt_pair in range(N_pairs):
                goal_state = env.random_valid_position()[0]
                init_state = env.reset()
                policy_Mr = GreedySMPolicy(env, Mr, goal_state)
                dist_ks_learn_td[i_d, k_cnt,data_cnt,cnt_pair] = env.test_policy_distance(policy_Mr, init_state, max_horizon = horizon)

dist_ks_learn_mb = np.zeros([5,len(k_arr),N_s,N_pairs])
for i_d in range(5):
    for data_cnt in range(N_s):
        print(data_cnt)
        dataset = env.generate_dataset(unif_pi, num_trajectories = num_trajs[data_cnt])
        _, P_pi = learn_SM_model_based(env, dataset, gamma=gamma, num_iters=1)
        for k_cnt in range(len(k_arr)):
            print(k_cnt)
            M = (1-gamma)*P_pi.copy()
            P_power = np.eye(dim)
            for t in range(horizon):
                P_power = P_power @ P_pi
                if t>=k_arr[k_cnt]:
                    M += (1-gamma)*(gamma ** (t-k_arr[k_cnt])) * P_power
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            Mr = utils.rank_r_approximation(U, S, Vt, r = rank)
            for cnt_pair in range(N_pairs):
                goal_state = env.random_valid_position()[0]
                init_state = env.reset()
                policy_Mr = GreedySMPolicy(env, Mr, goal_state)
                dist_ks_learn_mb[i_d,k_cnt,data_cnt,cnt_pair] = env.test_policy_distance(policy_Mr, init_state, max_horizon = horizon)

k_arr = [1, 3, 5, 9, 17]
xaxis = num_trajs

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [" ", " ", " ", "|", "1"]
linestyles = ["-", "--", "-.", "-", "-"]

for ax, dist_ks_learn in zip(axs, [dist_ks_learn_td, dist_ks_learn_mb]):
    for i, k in enumerate(k_arr):
        acc_ensamble = np.zeros([5, 10])
        for ell in range(5):
            acc_ensamble[ell, :] = np.mean(dist_ks_learn[ell, i, :, :] == 0, axis=1)

        accmean = np.mean(acc_ensamble, axis=0)
        accstd = np.std(acc_ensamble, axis=0)

        color = viridis(i)
        ax.plot(xaxis, accmean, label=f'$k={k}$', color=color,
                linestyle=linestyles[i], marker=markers[i], markevery=1)
        ax.fill_between(xaxis, accmean - accstd, accmean + accstd, color=color, alpha=0.2)

    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, 1e4)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.5)

axs[0].set_ylabel('accuracy', fontsize=18)
for ax in axs:
    ax.set_xlabel('number of trajectories', fontsize=18)

# Move legend to the right
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title='shift $k$',
    loc='center left',
    bbox_to_anchor=(0.95, 0.55),
)

fig.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on the right for the legend
plt.show()
