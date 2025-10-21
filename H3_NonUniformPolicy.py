import numpy as np
import matplotlib.pyplot as plt
from maze_environments import GridEnvironment
from policies import UnifRandomPolicy, AveragedGoalPolicy, GreedySMPolicy
from successor_measure import compute_occ_matrices_from_policy
import utils 
import matplotlib.cm as cm
import matplotlib as mpl

def nu_svd_r(M, nu, r):
    n = M.shape[0]
    assert M.shape == (n, n)
    assert nu.shape[0] == n
    assert np.all(nu > 0), "All nu values must be strictly positive"

    # Step 1: Define the reweighted matrix
    nu_sqrt = np.sqrt(nu)
    M_tilde = M * (nu_sqrt[:, None] / nu_sqrt[None, :])  # elementwise

    # Step 2: Standard SVD
    U, S, Vt = np.linalg.svd(M_tilde, full_matrices=False)

    V = Vt.T
    tU = np.zeros_like(U)
    tV = np.zeros_like(V)
    for x in range(np.shape(tU)[0]):
        for i in range(np.shape(tU)[1]):
            tU[x,i] = U[x,i] * np.sqrt(nu[i]/nu[x])
            tV[x,i] = V[x,i] * np.sqrt(nu[x]/nu[i])

    Ur = tU[:,0:r]
    Vr = tV[:,0:r]
    Sr = S[0:r]

    return Ur @ np.diag(Sr) @ Vr.T


env = GridEnvironment('Medium-maze',horizontal_exp = 2, vertical_exp = 2)
# pi = UnifRandomPolicy(env)
pi = AveragedGoalPolicy(env, num_goals = None, epsilon=0.8)

gamma = 0.95
N_pairs = 500
N_r = env.num_states
k_arr = [1, 3, 5, 9, 17]
horizon = 100
numS = env.num_states
numA = env.num_actions
dim = numS*numA
occ_matrices = compute_occ_matrices_from_policy(env, horizon=horizon, pi=pi)

valid_eye = np.zeros([dim,dim])
valid_states = np.zeros([dim]).astype(bool)
for i in range(dim):
    if occ_matrices[0][i,:].sum()>0:
        valid_eye[i,i] = 1 
        valid_states[i] = True
        
occ_matrices_full = [valid_eye] + occ_matrices

M = np.zeros_like(occ_matrices_full[0])
for t in range(0,horizon):
    M += (gamma ** t) * occ_matrices_full[t]  
    
Mp = M[np.ix_(valid_states, valid_states)]
nu = np.ones(Mp.shape[0]) / Mp.shape[0]
for _ in range(1000):
    nu = nu @ Mp*(1-gamma)
nu = nu / np.sum(nu)


dist_kr1 = np.zeros([len(k_arr),N_r,N_pairs])
dist_kr2 = np.zeros([len(k_arr),N_r,N_pairs])
norm_diff = np.zeros([len(k_arr),N_r])

for k_cnt in range(len(k_arr)):
    print(k_cnt)
    M = np.zeros_like(occ_matrices[0])
    for t in range(k_arr[k_cnt],horizon):
        M += (1-gamma)*(gamma ** (t-k_arr[k_cnt])) * occ_matrices_full[t]   
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    for r in range(N_r):
        print(r)
        Mr1 = utils.rank_r_approximation(U, S, Vt, r+1)
        
        Mp = M[np.ix_(valid_states, valid_states)]
        Mr2 = nu_svd_r(Mp, nu, r+1)
        Mr2full = np.zeros_like(M)
        Mr2full[np.ix_(valid_states, valid_states)] = Mr2
        norm_diff[k_cnt,r] = np.linalg.norm(Mr1-Mr2full)

        for cnt_pair in range(N_pairs):
            goal_state = env.random_valid_position()[0]
            init_state = env.reset()
            policy_Mr1 = GreedySMPolicy(env, Mr1, goal_state)
            dist_kr1[k_cnt,r,cnt_pair] = env.test_policy_distance(policy_Mr1, init_state, max_horizon = horizon)
            policy_Mr2 = GreedySMPolicy(env, Mr2full, goal_state)
            dist_kr2[k_cnt,r,cnt_pair] = env.test_policy_distance(policy_Mr2, init_state, max_horizon = horizon)
            

M = np.zeros_like(occ_matrices[0])
for t in range(0,horizon):
    M += (1-gamma)*(gamma ** (t)) * occ_matrices_full[t]   
norm_diff = norm_diff/np.linalg.norm(M)

# Set styles
import matplotlib.ticker as mtick
plt.figure(figsize=(8, 5))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})

viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [None, None, None, "|", "1", None]
linestyles = ["-", "--", "-.", "-", "-", "-"]

xaxis_r = np.arange(1, N_r + 1)


fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

for i, k in enumerate(k_arr):
    color = viridis(i)
    ax.plot(xaxis_r, 100 * norm_diff[i], label=f'$k={k}$',
            color=color, linestyle=linestyles[i], marker=markers[i], markevery=2, linewidth = 3)

ax.set_ylabel(r'$\frac{\Vert [M_k]_r - [M_k]_{r,\nu} \Vert_F }{\Vert M \Vert_F} $', fontsize=42, labelpad = 20)
ax.set_xlabel(r'rank $r$', fontsize=28)
ax.set_xlim(1, N_r)
ax.set_xticks([1] + list(np.arange(25, N_r + 1, 25)))
ax.set_yticks(np.arange(0, 26, 5))

# Format y-axis as percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

# Move legend outside
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='shift $k$', loc='center left',
           bbox_to_anchor=(0.7, 0.65), fontsize=22, title_fontsize=24, borderaxespad=0)




##############################################################################



plt.figure(figsize=(12, 5))
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})

xaxis_r = np.arange(1, len(dist_kr1[0, :, 0]) + 1)

viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)
markers = [" ", " ", " ", "|", "1", None]
linestyles = ["-", "--", "-.", "-", "-", "-"]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)


for subplot_id, dist_kr in enumerate([dist_kr1, dist_kr2]):
    ax = axs[subplot_id]
    for i, k in enumerate(k_arr):
        acc_ensamble = np.zeros([5, numS])
        for ell in range(5):
            acc_ensamble[ell, :] = np.mean(dist_kr[i, :, ell * 100:(ell + 1) * 100] == 0, axis=1)
        
        accmean = np.mean(acc_ensamble, axis=0)
        accstd = np.std(acc_ensamble, axis=0)

        color = viridis(i)
        ax.plot(xaxis_r, accmean, label=f'$k={k}$', color=color,
                linestyle=linestyles[i], marker=markers[i], markevery=10)
        ax.fill_between(xaxis_r, accmean - accstd, accmean + accstd, color=color, alpha=0.2)

    ax.set_ylim(0, 1.0)
    ax.set_xlim([1, numS])
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticks([1] + list(np.arange(25, numS, 25)))


# Set labels
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('rank')
axs[1].set_xlabel('rank')

# Move legend fully outside the plot area
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, title='shift $k$', loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=22, title_fontsize=24, borderaxespad=0)

plt.show()

##############################################################################

# Apply LaTeX and font size formatting
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(nu, color='tab:blue', linewidth=2)

# Vertical dashed line at 1/320 of the length of nu
x_pos = int(len(nu) * (1/320))
ax.axhline(y=1/320, color='gray', linestyle='--', linewidth=1)

# Axes labels and ticks
ax.set_xlabel('$(s,a)$ index', fontsize=28)
ax.set_ylabel(r'measure $\nu$', fontsize=28)
ax.set_xlim([0, len(nu) - 1])

# Optional: yticks from 0 to max value with 3 ticks
ax.set_yticks([0, 0.01])
ax.set_xticks([0,  300])

plt.tight_layout()
plt.show()
