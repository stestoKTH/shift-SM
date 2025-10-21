import numpy as np
import time
from copy import deepcopy
from collections import deque


class GridEnvironment:
    def __init__(self, maze_type, grid_size = None, cross_len = None, horizontal_exp = 1, vertical_exp = 1):
        self.maze_type = maze_type
        self.grid_size = grid_size
        self.cross_len = cross_len
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.expansion_factors = (vertical_exp, horizontal_exp)
        self.grid = self._create_grid()
        self.state_to_idx, self.idx_to_state = self._get_valid_state_indices()
        self.num_states = len(self.state_to_idx)
        self.num_actions = len(self.actions)
        self.all_distances = self._find_all_distances()


    def _create_grid(self):
        if self.maze_type == 'Cross-4' or self.maze_type == 'Cross-3':
            # Initialize grid with walls on the borders
            if self.grid_size is None:
                self.grid_size = (9,9)
            grid = np.zeros((self.grid_size[0], self.grid_size[0]), dtype=int)  # 0 = free, 1 = wall
            grid[0, :] = grid[-1, :] = 1  # Top and bottom walls
            grid[:, 0] = grid[:, -1] = 1  # Left and right walls
            
            # Add central horizontal and vertical walls (splitting into four rooms)
            center = (self.grid_size[0]-1) // 2
            
            if self.cross_len is None:
                grid[center, :] = 1  # Horizontal wall
                grid[:, center] = 1  # Vertical wall
                # Create doors at random positions on the walls
                doors = np.random.randint(1, center, size=4)  # One door per wall
                grid[center, doors[0]] = 0                    # Left door
                grid[center, center + doors[1]] = 0           # Right door
                grid[doors[2], center] = 0                    # Top door
                grid[center + doors[3], center] = 0           # Bottom door
            else:
                # Vertical spike of the cross
                for i in range(-self.cross_len, self.cross_len + 1):
                    grid[center + i, center] = 1
                # Horizontal spike of the cross
                for j in range(-self.cross_len, self.cross_len + 1):
                    grid[center, center + j] = 1
            
            if self.maze_type == 'Cross-4':
                return grid       
            else:   
                # Modify the grid to create a three-room environment (remove part of one wall)
                grid[center, center + 1:-1] = 0  # Open right half of the horizontal wall
                return grid
            
        elif self.maze_type == 'U-maze':
            maze_array = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
            self.grid_size = tuple(x * y for x, y in zip((5,5), self.expansion_factors))
            
        elif self.maze_type == 'Medium-maze':
            maze_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
            self.grid_size = tuple(x * y for x, y in zip((8,8), self.expansion_factors))
            
        elif self.maze_type == 'Large-maze':
            maze_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
            self.grid_size = tuple(x * y for x, y in zip((9,12), self.expansion_factors))
            
        else:
            raise ValueError("Invalid maze type. Use Cross-4, Cross-3, U-maze, Medium-maze or Large-maze")

        return np.repeat(np.repeat(maze_array, self.expansion_factors[1], axis=1), self.expansion_factors[0], axis=0)


    def _get_valid_state_indices(self):
        state_to_idx = {}
        idx_to_state = {}
        idx = 0
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid[x, y] == 0:
                    state_to_idx[(x, y)] = idx
                    idx_to_state[idx] = (x, y)
                    idx += 1
        return state_to_idx, idx_to_state
    
    def random_valid_position(self, num = 1):
        empty_cells = np.argwhere(self.grid == 0)
        if empty_cells.size == 0:
            raise ValueError("No empty cells available.")
        idx = np.random.choice(len(empty_cells), size=num, replace=False)
        return [tuple(empty_cells[i]) for i in idx]                    
            
    def reset(self, locs = None):
        if locs is not None:
            return locs
        else:
            return self.random_valid_position()[0]


    def step(self, state, action):
        if action not in self.actions:
            raise ValueError("Invalid action. Use 0=up, 1=down, 2=left, 3=right.")
        is_valid, next_state = self.is_valid_transition(state, action)
        if is_valid:
            return True, next_state
        else:
            return False, state
            
                
    def sa_index(self, s, a):
        return self.state_to_idx[s] * len(self.actions) + a

            
    def index_to_sa(self, idx):
        s_idx = idx // self.len(self.actions)
        a = idx % self.len(self.actions)
        s = self.idx_to_state[s_idx]
        return s, a
            
    
    def is_valid_transition(self, s, a):
        dx, dy = self.actions[a]
        x, y = s
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
            return self.grid[nx, ny] == 0, (nx,ny)
        return False, None


    def render_grid(self, agent_state):
        display = np.array(self.grid, dtype=str)
        display[self.grid == 0] = "."
        display[self.grid == 1] = "#"
        x, y = agent_state
        display[x, y] = "A"
        print("\n".join(" ".join(row) for row in display))


    def render_policy(self, policy, steps=50, sleep=0.2, initial_state = None):
        state = self.reset(locs = initial_state)
        for step in range(steps):
            display = deepcopy(self.grid).astype(str)
            display[self.grid == 0] = "."
            display[self.grid == 1] = "#"
    
            gx, gy = policy.goal if policy.goal else (-1, -1)
            if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                display[gx, gy] = "G"
    
            # x, y = state
            display[state] = "A"
    
            print(f"\nStep {step + 1}/{steps}")
            print("\n".join(" ".join(row) for row in display))
            print(f"Goal: {policy.goal}\n")
    
            time.sleep(sleep)
            # import pdb; pdb.set_trace()
            action, valid_flag = policy.act(state)
            if state == policy.goal or not valid_flag:
                break
            _, state = self.step(state, action)
            
            
    def generate_dataset(self, policy, num_trajectories=10000, max_steps=100):
        dataset = []
        for _ in range(num_trajectories):
            trajectory = []
            state = self.reset()
            for _ in range(max_steps):
                action, _ = policy.act(state)
                _, next_state = self.step(state, action)
                # trajectory.append((state, action, next_state))
                trajectory.append((state, action))
                state = next_state
            dataset.append(trajectory)
        return dataset
    
    
    def _find_all_distances(self):
        all_distances = np.zeros([self.num_states, self.num_states])
        for state_idx in range(self.num_states):
            for goal_idx in range(self.num_states):
                visited = set()
                queue = deque([(state_idx, 0)])  # (current_state, distance_so_far)
    
                while queue:
                    current_idx, dist = queue.popleft()
                    if current_idx == goal_idx:
                        all_distances[state_idx, goal_idx] = dist
                        break  # stop BFS once shortest path is found
    
                    if current_idx in visited:
                        continue
                    visited.add(current_idx)
    
                    for a in self.actions:
                        try:
                            valid_step, neighbor = self.is_valid_transition(self.idx_to_state[current_idx], a)
                            if valid_step:
                                neighbor_idx = self.state_to_idx[neighbor]
                        except Exception as e:
                            print(f"Error with state index {current_idx}, action {a}: {e}")                            
                        if valid_step and neighbor_idx not in visited:
                            queue.append((neighbor_idx, dist + 1))
        return all_distances
    
    
    def test_policy_distance(self, policy, init_state, max_horizon = 30):
        state = init_state
        for step in range(max_horizon):  
            if state == policy.goal:
                return 0
            action, valid_flag = policy.act(state)
            if valid_flag:
                _, state = self.step(state, action)
            else:
                print(f"No valid action for {state}")                            
        return self.all_distances[self.state_to_idx[state], self.state_to_idx[policy.goal]]
        