import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def two_to_infinity_norm(A):
    row_norms = np.linalg.norm(A, axis=1)
    return np.max(row_norms)

def xi_custom(M):
    """
    Computes ξ(M) = max_x ∑_i σ_i * ψ_i(x)^2
    where ψ_i(x) are entries of the left singular vectors of M.
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)  # M = U Σ Vᵀ
    weighted = np.zeros(len(S))
    for i in range(len(S)):
        weighted[i] = np.dot(U[i, :]**2, S)  

    return np.max(weighted)


def matrix_sqrt_svd(A):
    # Perform SVD: A = U * S * V^T
    U, S, Vt = np.linalg.svd(A)
    
    # Take square root of singular values
    S_sqrt = np.sqrt(S)
    
    # Reconstruct A^{1/2}: U * sqrt(S) * V^T
    A_sqrt = U @ np.diag(S_sqrt) @ Vt
    return A_sqrt

def rank_r_approximation(U, S, Vt, r):
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    Vt_r = Vt[:r, :]
    return U_r @ S_r @ Vt_r



def plot_boxes_SM(env, M, goal):
    goal_idx = env.state_to_idx[goal]
    M_averaged_actions = M.reshape(M.shape[0]//4, 4, M.shape[1]//4, 4).mean(axis=(1, 3))
    values = M_averaged_actions[:, goal_idx]  # shape (num_states,)

    # Create an empty 2D grid of the right shape
    grid_shape = env.grid_size  # or manually set (rows, cols)
    value_grid = np.full(grid_shape, np.nan)

    # Fill the value grid using idx_to_state
    for idx in range(values.shape[0]):
        state = env.idx_to_state[idx]  # returns (row, col)
        value_grid[state] = values[idx]

    # Plot
    plt.imshow(value_grid, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(f'Values toward goal {goal}')
    plt.show()


def plot_triangles_SM(env, M, goal):
    goal_idx = env.state_to_idx[goal]*4 + np.arange(0,4,1)
    values = np.mean(M[:, goal_idx], axis=1)  # shape: (num_sa,)
    
    # Number of states = num_sa / 4 if 4 actions per state
    num_actions = env.num_actions
    num_states = env.num_states
    
    # Reshape to (num_states, 4) — one row per state, with 4 action values
    state_action_values = values.reshape((num_states, num_actions))
    
    # Create patches
    patches = []
    colors = []
    
    for s_idx in range(num_states):
        r, c = env.idx_to_state[s_idx]  # get row/col of this state
    
        cell_center = (c + 0.5, r + 0.5)
        corners = [
            (c, r),       # top-left
            (c + 1, r),   # top-right
            (c + 1, r + 1), # bottom-right
            (c, r + 1),   # bottom-left
        ]
    
        # Define triangles for actions 0 (up), 1 (right), 2 (down), 3 (left)
        triangles = [
            [corners[0], corners[1], cell_center],  # up
            [corners[2], corners[3], cell_center],  # down
            [corners[3], corners[0], cell_center],  # left
            [corners[1], corners[2], cell_center],  # right
        ]
    
        for a in range(num_actions):
            patches.append(Polygon(triangles[a]))
            colors.append(state_action_values[s_idx, a])
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    p = PatchCollection(patches, cmap='viridis', edgecolor='none')
    p.set_array(np.array(colors))
    ax.add_collection(p)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.autoscale_view()
    fig.colorbar(p, ax=ax, label='Value')
    plt.title(f'Per-action values toward goal {goal}')
    plt.show()
    
    
    
def plot_triangles_SM_reversed(env, M, goal):
    goal_idx = env.state_to_idx[goal]*4 + np.arange(0,4,1)
    values = np.mean(M[:, goal_idx], axis=1)  # shape: (num_sa,)
    
    # Number of states = num_sa / 4 if 4 actions per state
    num_actions = env.num_actions
    num_states = env.num_states
    
    # Reshape to (num_states, 4) — one row per state, with 4 action values
    state_action_values = values.reshape((num_states, num_actions))
    
    action_to_reverse_triangle = {
    0: 2,  # up → show in bottom triangle of neighbor
    1: 0,  # down → show in top triangle of neighbor
    2: 1,  # left → show in right triangle of neighbor
    3: 3,  # right → show in left triangle of neighbor
    }
    
    action_to_forward_triangle = {  # to draw in current state when invalid
    0: 0,  # up → top
    1: 2,  # bottom
    2: 3,  # left
    3: 1,  # right
    }
    
    patches = []
    colors = []
    
    for s_idx in range(num_states):
        x, y = env.idx_to_state[s_idx]
    
        for a in range(num_actions):
            value = state_action_values[s_idx, a]
    
            valid, (nx, ny) = env.is_valid_transition((x, y), a)
            if valid:
                cx, cy = nx, ny
                tri_idx = action_to_reverse_triangle[a]
                color = value
            else:
                cx, cy = x, y
                tri_idx = action_to_forward_triangle[a]
                color = 0.0
    
            cell_center = (cy + 0.5, cx + 0.5)
            corners = [
                (cy, cx), (cy + 1, cx), (cy + 1, cx + 1), (cy, cx + 1)
            ]
    
            triangles = [
                [corners[0], corners[1], cell_center],  # top
                [corners[1], corners[2], cell_center],  # right
                [corners[2], corners[3], cell_center],  # bottom
                [corners[3], corners[0], cell_center],  # left
            ]
    
            patches.append(Polygon(triangles[tri_idx]))
            colors.append(color)
            

    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Patch collection with colors
    p = PatchCollection(patches, cmap='viridis', edgecolor='none')
    p.set_array(np.array(colors))
    ax.add_collection(p)
    
    # Goal marker (a star)
    goal_y, goal_x = goal  # assuming goal = (row, col)
    ax.plot(goal_x + 0.5, goal_y + 0.5, marker='*', color='red', markersize=10, markeredgecolor='black', zorder=10)
    
    # Axis formatting
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
        
    plt.tight_layout()
    plt.show()
    
    
    
    



