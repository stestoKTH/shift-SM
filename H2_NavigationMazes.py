import numpy as np
import matplotlib.pyplot as plt
from maze_environments import GridEnvironment
from policies import UnifRandomPolicy, GreedySMPolicy
from successor_measure import compute_occ_matrices_from_policy, learn_SM_td
import utils
import matplotlib.cm as cm
import matplotlib as mpl

# %%       ENV + Sing. values

exp_factor = 2 # 4 for U, 2 for Medium and Large
env = GridEnvironment('Medium-maze',horizontal_exp = exp_factor, vertical_exp = exp_factor)
# env = GridEnvironment('Large-maze',horizontal_exp = exp_factor, vertical_exp = exp_factor)
# env = GridEnvironment('U-maze',horizontal_exp = exp_factor, vertical_exp = exp_factor)
pi = UnifRandomPolicy(env)
horizon = 200
occ_matrices = compute_occ_matrices_from_policy(env, horizon=horizon, pi=pi)
gamma = 0.95 # 0.98 for U and Large, 0.95 for Medium
numS = env.num_states
numA = env.num_actions


k_arr = [0, 2, 4, 8, 16]

singular_values = np.zeros([len(k_arr),numS*numA])
for ind in range(len(k_arr)):
    M = np.zeros_like(occ_matrices[0])
    for t in range(k_arr[ind], len(occ_matrices)):
        M += (gamma ** (t - k_arr[ind])) * occ_matrices[t]
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    singular_values[ind,:] = S[:]
    


###############
maze_crop = exp_factor-1
goal = (4 + maze_crop, 1 + maze_crop) # medium (4,1), U (3,12), Large (11,11)
goal_idx = env.state_to_idx[goal]

M = np.zeros_like(occ_matrices[0])
for t in range(0, len(occ_matrices)):
    M += (gamma ** (t)) * occ_matrices[t]


M_av_max_actions = M.reshape(M.shape[0]//4, 4, M.shape[1]//4, 4).mean(axis=3).max(axis=1)
values = M_av_max_actions[:, goal_idx]  # shape (num_states,)

# Create an empty 2D grid of the right shape
grid_shape = env.grid_size  # or manually set (rows, cols)
value_grid = np.full(grid_shape, np.nan)

action_to_vec = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),   # left
    3: (0, 1),   # right
}


M_goal_av = M.reshape(M.shape[0], M.shape[1]//4, 4).mean(axis=2)

M_sa_goal = M_goal_av[:, goal_idx].reshape(-1, 4)  # shape: (num_states, 4)
best_actions = np.argmax(M_sa_goal, axis=1)  # shape: (num_states,)


# Fill the value grid using idx_to_state
for idx in range(values.shape[0]):
    state = env.idx_to_state[idx]  # returns (row, col)
    value_grid[state] = values[idx]

k_arr = [1, 3, 5, 9, 17]
viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)

X, Y, U, V = [], [], [], []
for idx in range(len(best_actions)):
    row, col = env.idx_to_state[idx]
    dx, dy = action_to_vec[best_actions[idx]]
    X.append(col - maze_crop )
    Y.append(row - maze_crop)
    U.append(0.6 * dy)
    V.append(0.6 * dx)


# --- Create combined figure ---
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})
fig, axs = plt.subplots(2, 1, figsize=(6,9), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.2})

# --- Bottom plot: Singular value decay ---
ax = axs[1]
for i, k in enumerate(k_arr):
    ax.scatter(np.arange(1, numS+1), singular_values[i, :numS], label=f'$k={k}$', color=viridis(i), s=10)

ax.set_yscale('log')
ax.set_ylim(1e-5*20, 20)
ax.set_xlim(1, numS)
ax.set_xlabel('index $i$', fontsize=30)
ax.set_ylabel('$\sigma_i(M_{\pi,k})$', fontsize=30)
ax.grid(True, which='major', linestyle='--', linewidth=0.25)
ax.set_xticks([1] + list(np.arange(25, numS, 25)))
ax.set_yticks([1e-4, 1e-2, 1, 100])
axs[1].text(0.88, 0.95, '(b)', transform=axs[1].transAxes, fontsize=26, va='top')

# --- Top plot: Maze with arrows ---

ax = axs[0]
# ax.set_facecolor("#b07c4c")
ax.set_facecolor("gray")
im = ax.imshow(value_grid[maze_crop:-maze_crop, maze_crop:-maze_crop], cmap='Oranges_r', origin='upper')
goal_y, goal_x = goal
ax.plot(goal_x - maze_crop, goal_y - maze_crop+0.2, marker='*', color='white', markersize=15, markeredgecolor='black', zorder=5)
ax.quiver(X, Y, U, V, color='black', scale=1, scale_units='xy', angles='xy', linewidth=5)
ax.quiver(X, Y, U, V, color='white', scale=1, scale_units='xy', angles='xy', linewidth=1)

rows, cols = value_grid[maze_crop:-maze_crop, maze_crop:-maze_crop].shape
for r in range(rows + 4):
    ax.axhline(r - 0.5, color='white', linewidth=0.5)
for c in range(cols + 4):
    ax.axvline(c - 0.5, color='white', linewidth=0.5)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-0.5, cols - 0.5])
ax.set_ylim([rows - 0.5, -0.5])  # flip y-axis to match image origin


from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("bottom", size="5%", pad=0.1)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=24)
# cb.ax.set_xticks([0, 0.5, 1])


# --- Final layout and display ---
axs[0].text(-0.15, 0.88, '(a)', transform=axs[0].transAxes, fontsize=26, va='bottom')
plt.tight_layout()

plt.show()






#%%     M exact experiment

# env = GridEnvironment('Cross-4', grid_size = [15,15], cross_len = 5)
env = GridEnvironment('Medium-maze',horizontal_exp = 2, vertical_exp = 2)
# env = GridEnvironment('Large-maze',horizontal_exp = 2, vertical_exp = 2)
# env = GridEnvironment('U-maze',horizontal_exp = 4, vertical_exp = 4)

unif_pi = UnifRandomPolicy(env)
# horizon = 300
horizon = 100
numS = env.num_states
numA = env.num_actions
dim = numS*numA

occ_matrices = compute_occ_matrices_from_policy(env, horizon=horizon, pi=unif_pi)

gamma = 0.95 # 0.95 for 9x9, 0.97 for 15x15
k_arr = [0, 2, 4, 8, 16]

N_pairs = 500
N_r = env.num_states

dist_kr = np.zeros([len(k_arr),N_r,N_pairs])
for k_cnt in range(len(k_arr)):
    print(k_cnt)
    M = np.zeros_like(occ_matrices[0])
    for t in range(k_arr[k_cnt],horizon):
        M += (1-gamma)*(gamma ** (t-k_arr[k_cnt])) * occ_matrices[t]   
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    for r in range(N_r):
        print(r)
        Mr = utils.rank_r_approximation(U, S, Vt, r+1)
        for cnt_pair in range(N_pairs):
            goal_state = env.random_valid_position()[0]
            init_state = env.reset()
            policy_Mr = GreedySMPolicy(env, Mr, goal_state)
            dist_kr[k_cnt,r,cnt_pair] = env.test_policy_distance(policy_Mr, init_state, max_horizon = horizon)
        


plt.figure(figsize=(6, 4))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})
xaxis_r = np.arange(1,len(dist_kr[0,:,0])+1)
k_arr = [1, 3, 5, 9, 17]

acc_threshold = [0, 2]

viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [" "," ",  " ", "|", "1"]
linestyles = ["-", "--", "-.", "-", "-"]

fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True, sharey=True)

for subplot_id in range(2):  # one plot per column
    ax = axs[subplot_id]

    for i, k in enumerate(k_arr):
        acc_ensamble = np.zeros([5, numS])
        for ell in range(5):
            acc_ensamble[ell,:] = np.mean(dist_kr[i, :, ell*100:(ell+1)*100] <= acc_threshold[subplot_id], axis=1)
            
        accmean = np.mean(acc_ensamble, axis=0)
        accstd = np.std(acc_ensamble, axis=0)
    
        color = viridis(i)
        ax.plot(xaxis_r, accmean, label=f'$k={k}$', color=color, linestyle = linestyles[i],  marker=markers[i], markevery=10)
        ax.fill_between(xaxis_r, accmean - accstd, accmean + accstd, color=color, alpha=0.2)  

    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([1] + list(np.arange(25, numS, 25)))
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim([1,numS])
    
axs[1].set_xlabel('rank', fontsize=34)
axs[0].tick_params(labelbottom=True)
axs[0].set_ylabel('accuracy', fontsize=30, labelpad=34)
axs[1].set_ylabel('relaxed accuracy', fontsize=30, labelpad=34)

fig.subplots_adjust(left=0.25, hspace=0.2)  # increase left margin only
# fig.tight_layout()
axs[0].text(0.02, 0.95, '(c)', transform=axs[0].transAxes, fontsize=26, va='top')
axs[1].text(0.02, 0.95, '(d)', transform=axs[1].transAxes, fontsize=26, va='top')
plt.show()


# %%          M TD experiment

# env = GridEnvironment('Medium-maze',horizontal_exp = 2, vertical_exp = 2)
env = GridEnvironment('Large-maze',horizontal_exp = 2, vertical_exp = 2)
# env = GridEnvironment('U-maze',horizontal_exp = 4, vertical_exp = 4)
unif_pi = UnifRandomPolicy(env)
N_r = env.num_states
N_pairs = 100
gamma = 0.99
horizon = 300

k_arr = [0, 2, 4, 8, 16]
num_rep = 5
dist_kr_learn_td = np.zeros([num_rep,len(k_arr),N_r,N_pairs])

for i_d in range(num_rep):
    dataset = env.generate_dataset(unif_pi)
    for k_cnt in range(len(k_arr)):
        print(k_cnt)
        M = learn_SM_td(env, dataset, gamma=gamma, num_samples=int(1e5), alpha=0.1, k_shift=k_arr[k_cnt])
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        for r in range(N_r):
            Mr = utils.rank_r_approximation(U, S, Vt, r = r+1)
            for cnt_pair in range(N_pairs):
                goal_state = env.random_valid_position()[0]
                init_state = env.reset()
                policy_Mr = GreedySMPolicy(env, Mr, goal_state)
                dist_kr_learn_td[i_d, k_cnt,r,cnt_pair] = env.test_policy_distance(policy_Mr, init_state, max_horizon = horizon)
            


numS = env.num_states
plt.figure(figsize=(6, 4))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})
xaxis_r = np.arange(1,len(dist_kr_learn_td[0,0,:,0])+1)
k_arr = [1, 3, 5, 9, 17]

acc_threshold = [0, 2]
viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [" "," ",  " ", "|", "1"]
linestyles = ["-", "--", "-.", "-", "-"]

fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharey=True)

for subplot_id in range(2):  # one plot per column
    ax = axs[subplot_id]

    for i, k in enumerate(k_arr):
        acc_ensamble = np.zeros([num_rep, numS])
        for ell in range(num_rep):
            acc_ensamble[ell,:] = np.mean(dist_kr_learn_td[ell, i, :, :] <= acc_threshold[subplot_id], axis=1)
            
        accmean = np.mean(acc_ensamble, axis=0)
        accstd = np.std(acc_ensamble, axis=0)
    
        color = viridis(i)
        ax.plot(xaxis_r, accmean, label=f'$k={k}$', color=color, linestyle = linestyles[i],  marker=markers[i], markevery=10)
        ax.fill_between(xaxis_r, accmean - accstd, accmean + accstd, color=color, alpha=0.2)  

    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([1] + list(np.arange(25, numS, 25)))
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlim([1,numS])

axs[1].set_xlabel('rank', fontsize=34)
fig.tight_layout()
axs[0].text(0.02, 0.95, '(e)', transform=axs[0].transAxes, fontsize=26, va='top')
axs[1].text(0.02, 0.95, '(f)', transform=axs[1].transAxes, fontsize=26, va='top')
plt.show()


# %%        Sample complexity experiment

# env = GridEnvironment('Medium-maze',horizontal_exp = 2, vertical_exp = 2)
# env = GridEnvironment('Large-maze',horizontal_exp = 2, vertical_exp = 2)
env = GridEnvironment('U-maze',horizontal_exp = 4, vertical_exp = 4)

unif_pi = UnifRandomPolicy(env)
num_steps = 10
num_trajs = np.round(np.logspace(0.0, 4.0, num=num_steps)).astype(int)  
N_s = len(num_trajs)
rank = 40 # 60 for L, 40 for M and U 
dim = env.num_states*env.num_actions
N_pairs = 100
gamma = 0.98 # 0.95 for M, 0.98 for L and U
horizon = 100
k_arr = [0, 2, 4, 8, 16]
num_rep = 5

dist_ks_learn_td = np.zeros([num_rep,len(k_arr),N_s,N_pairs])

for i_d in range(num_rep):
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
          

plt.figure(figsize=(6, 4))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})
xaxis = num_trajs
k_arr = [1, 3, 5, 9, 17]

acc_threshold = [0, 2]
viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [" "," ",  " ", "|", "1"]
linestyles = ["-", "--", "-.", "-", "-"]

fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharey=True)

for subplot_id in range(2):  # one plot per column
    ax = axs[subplot_id]

    for i, k in enumerate(k_arr):
        acc_ensamble = np.zeros([num_rep, num_steps])
        for ell in range(num_rep):
            acc_ensamble[ell,:] = np.mean(dist_ks_learn_td[ell, i, :, :] <= acc_threshold[subplot_id], axis=1)
            
        accmean = np.mean(acc_ensamble, axis=0)
        accstd = np.std(acc_ensamble, axis=0)
    
        color = viridis(i)
        ax.plot(xaxis, accmean, label=f'$k={k}$', color=color, linestyle = linestyles[i],  marker=markers[i], markevery=1)
        ax.fill_between(xaxis, accmean - accstd, accmean + accstd, color=color, alpha=0.2)  

    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, 1e4)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(0.75, color='gray', linestyle='--', linewidth=0.5)

# axs[0].set_ylabel('accuracy')
axs[1].set_xlabel('number of trajectories', fontsize=34)

fig.tight_layout()
axs[0].text(0.02, 0.95, '(g)', transform=axs[0].transAxes, fontsize=26, va='top')
axs[1].text(0.02, 0.95, '(h)', transform=axs[1].transAxes, fontsize=26, va='top')
plt.show()


# %%    LEGEND

fig, ax = plt.subplots(figsize=(6, 1))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 28})
lines = []

for i, k in enumerate(k_arr):
    line, = ax.plot([], [],  
                    label=f'$k={k}$',
                    color=viridis(i),
                    marker=markers[i],
                    linestyle=linestyles[i])
    lines.append(line)

ax.axis('off')


legend = ax.legend(
    handles=[plt.Line2D([0], [0], color='none', label='$k$-shift')] + lines,
    loc='center',
    ncol=len(k_arr) + 1,  
    fontsize=34,
    frameon=True,          
    handlelength=2,
    handletextpad=0.5,
    columnspacing=1.2,
    borderpad=0.5
)






