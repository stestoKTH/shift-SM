import numpy as np
import matplotlib.pyplot as plt
from maze_environments import GridEnvironment
from policies import UnifRandomPolicy
import utils 
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors



n = 15
env = GridEnvironment('Cross-4', grid_size = [n,n], cross_len = int((n-1)/2-2))
horizon = 100
gamma = 0.97
numS = env.num_states
numA = env.num_actions
dim = numS*numA


##############################################################################


value_grid = env.grid  # assumed to contain 0s and 1s

# Define custom colormap: 0 → light blue, 1 → gray
cmap = mcolors.ListedColormap(['lightgray', 'gray'])  # light blue, gray
bounds = [-0.5, 0.5, 1.5]  # so that 0 maps to first color, 1 to second
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_facecolor("gray")  # background outside the image
ax.imshow(value_grid, cmap=cmap, norm=norm, origin='upper')

# Grid overlay
rows, cols = value_grid.shape
for r in range(rows + 1):
    ax.axhline(r - 0.5, color='white', linewidth=1)
for c in range(cols + 1):
    ax.axvline(c - 0.5, color='white', linewidth=1)

# Remove ticks and adjust axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-0.5, cols - 0.5])
ax.set_ylim([rows - 0.5, -0.5])  # Flip y-axis to match image origin


plt.tight_layout()
plt.show()


##############################################################################


pi = UnifRandomPolicy(env)
P_pi = np.zeros((dim, dim))  # Transition matrix under policy π
# Build P_pi[(s,a), (s',a')]
for s in env.state_to_idx:
    for a in env.actions:
        valid, s_prime = env.is_valid_transition(s, a)
        if not valid:
            continue
        sa_idx = env.sa_index(s, a)
        for a_prime in env.actions:
            if env.is_valid_transition(s_prime, a_prime)[0]:
                prob = pi.pi_probability(s_prime, a_prime)
                sa_prime_idx = env.sa_index(s_prime, a_prime)
                P_pi[sa_idx, sa_prime_idx] += prob


valid_eye = np.zeros([dim,dim])
valid_states = np.zeros([dim]).astype(bool)
for i in range(dim):
    if P_pi[i,:].sum()>0:
        valid_eye[i,i] = 1 
        valid_states[i] = True

P_pi = (P_pi + 0.1*valid_eye)/1.1


k_arr = [0, 2, 5, 10, 20]
singular_values = np.zeros([len(k_arr),numS*numA])
for ind in range(len(k_arr)):
    
    M = np.zeros([dim,dim])
    tmp_power = np.eye(dim)
    for ind2 in range(k_arr[ind]):
        tmp_power = tmp_power @ P_pi
    for ind22 in range(k_arr[ind], horizon):
        M += tmp_power
        tmp_power = gamma*tmp_power @ P_pi
        
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    singular_values[ind,:] = S[:]
    


num_sing_values = 8

mpl.rcParams['text.usetex'] = True
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 20})
viridis = cm.get_cmap('Blues_r', len(k_arr) + 2)

for i, k in reversed(list(enumerate(k_arr))):
    plt.scatter(np.arange(1, num_sing_values + 1), singular_values[i, 0:num_sing_values], 
                label=f'$k={k}$', color=viridis(i), s=40)
    
# Add horizontal line and text
plt.axhline(singular_values[0,0], color='black', linestyle=':', linewidth=1)
plt.text(0.8, singular_values[0,0], r'$(1 - \gamma)^{-1}$', 
         fontsize=18, va='center', ha='right')

plt.yscale('log')
plt.ylim(1, 1.1/(1-gamma))
plt.xlim(1-0.1, num_sing_values+0.1)

plt.xlabel('index $i$', fontsize=20)
plt.ylabel(r'$\sigma_i(M_{\pi,k})$', fontsize=20, labelpad=20)
plt.grid(True, which='major', linestyle='--', linewidth=0.25)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='shift $k$', fontsize=16, title_fontsize=16)

plt.tight_layout()
plt.show()



##############################################################################
      

k_arr2 = np.arange(0,21,2)
err_n15 = np.zeros(len(k_arr2))
err_n15M = np.zeros(len(k_arr2))

for ind_k, k in enumerate(k_arr2):
    tmp_mat = np.eye(np.shape(P_pi)[0])
    for _ in range(0,k):
        tmp_mat = tmp_mat @ P_pi
        
    U, S, Vt = np.linalg.svd(tmp_mat, full_matrices=False)
    tmp_rmat = utils.rank_r_approximation(U, S, Vt, r = 4)
    err_n15[ind_k] = utils.two_to_infinity_norm(tmp_mat-tmp_rmat)   
    
    tmp_M = np.zeros([dim,dim])
    tmp_power = np.eye(dim)
    for ind2 in range(k):
        tmp_power = tmp_power @ P_pi
    for ind22 in range(k, horizon):
        tmp_M += tmp_power
        tmp_power = gamma*tmp_power @ P_pi

    U, S, Vt = np.linalg.svd(tmp_M, full_matrices=False)
    tmp_rM = utils.rank_r_approximation(U, S, Vt, r = 4)
    err_n15M[ind_k] = utils.two_to_infinity_norm(tmp_M-tmp_rM)   
    
    

plt.rc('text', usetex=True)
plt.rc('font', size=26)
fig, ax1 = plt.subplots(figsize=(8, 5))

# Left y-axis for err_n15
color1 = 'tab:blue'
ax1.set_xlabel('shift $k$', fontsize=26)
ax1.set_ylabel('$ \Vert P^k - [P^k]_4 \Vert_{2,\infty}$', color=color1,fontsize=26)
ax1.plot(k_arr2, err_n15, color=color1, marker='o', label=None)
ax1.tick_params(axis='y', labelcolor=color1)

# Right y-axis for err_n15M
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('$ \Vert M_{\pi,k} - [M_{\pi,k}]_4 \Vert_{2,\infty}$', color=color2, fontsize=26)
ax2.plot(k_arr2, err_n15M, color=color2, marker='s', label=None)
ax2.tick_params(axis='y', labelcolor=color2)

# Title and grid
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

