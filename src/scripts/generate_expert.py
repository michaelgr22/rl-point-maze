import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
import heapq
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ENV_ID = "PointMaze_Medium-v3"
TOTAL_EPISODES = 1000        # Set this to how many episodes you want to record
OUTPUT_FILE = "expert_data_hires.pkl"
SHOW_VISUALIZATION = True # Set True to watch the agent live
RESOLUTION_SCALE = 10     # High resolution for smooth paths

MANUAL_OFFSET_X = -0.5 # Try -1.2 first (shifts grid Center Left)
MANUAL_OFFSET_Y = -0.4  # Slight adjustment Up might be needed too

# --- 1. COORDINATE MAPPING ---
def world_to_grid(x, y, shape):
    rows, cols = shape 
    off_x = (cols // 2) + 0.5 + MANUAL_OFFSET_X
    off_y = (rows // 2) + 0.5 + MANUAL_OFFSET_Y
    
    scaled_off_x = off_x * RESOLUTION_SCALE
    scaled_off_y = off_y * RESOLUTION_SCALE
    scaled_x = x * RESOLUTION_SCALE
    scaled_y = y * RESOLUTION_SCALE

    row = int(round(scaled_off_y - scaled_y - 0.5))
    col = int(round(scaled_off_x + scaled_x - 0.5))
    
    max_r = (rows * RESOLUTION_SCALE) - 1
    max_c = (cols * RESOLUTION_SCALE) - 1
    return max(0, min(row, max_r)), max(0, min(col, max_c))

# --- ADD THIS HELPER ---
def world_to_grid_float(x, y, shape):
    """
    Same as world_to_grid, but returns EXACT float coordinates (row, col)
    instead of rounding to integers. This reveals sub-pixel misalignment.
    """
    rows, cols = shape 
    off_x = (cols // 2) + 0.5+ MANUAL_OFFSET_X
    off_y = (rows // 2) + 0.5+ MANUAL_OFFSET_Y
    
    scaled_x = x * RESOLUTION_SCALE
    scaled_y = y * RESOLUTION_SCALE
    
    # Precise float calculation
    row = (off_y * RESOLUTION_SCALE) - scaled_y - 0.5
    col = (off_x * RESOLUTION_SCALE) + scaled_x - 0.5
    
    return row, col

# --- UPDATED DEBUG VISUALIZER ---
def visualize_debug(env, hires_map, safe_map, start_node, goal_node, path_nodes, start_xy):
    real_img = env.render()
    
    # Calculate the raw floating-point grid position
    raw_r, raw_c = world_to_grid_float(start_xy[0], start_xy[1], (5, 5)) # 5x5 is original shape for Medium? 
    # NOTE: Pass 'original_shape' variable here in main loop, likely (rows, cols) of raw_map
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Reality
    axes[0].imshow(real_img)
    axes[0].set_title("Reality (MuJoCo)")
    axes[0].axis('off')
    
    # Right: The Planner's Brain
    axes[1].imshow(safe_map, cmap='Greys', origin='upper')
    
    # Plot Path
    if path_nodes:
        rs = [p[0] for p in path_nodes]
        cs = [p[1] for p in path_nodes]
        axes[1].plot(cs, rs, c='blue', linewidth=2, alpha=0.5, label="A* Path")
        
    # 1. Plot the Snapped Integer Node (What A* uses)
    axes[1].scatter(start_node[1], start_node[0], c='green', s=150, label="Snapped Node (Integer)")
    
    # 2. Plot the TRUE Continuous Position (Where physics puts it)
    axes[1].scatter(raw_c, raw_r, c='gold', marker='x', s=150, linewidth=3, label="True Raw Pos (Float)")
    
    axes[1].scatter(goal_node[1], goal_node[0], c='red', s=150, marker='*', label="Goal")
    
    axes[1].set_title(f"Alignment Check\nYellow X inside Wall = BAD OFFSET")
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


def grid_to_world(r, c, shape):
    rows, cols = shape
    off_x = ((cols // 2) + 0.5) * RESOLUTION_SCALE
    off_y = ((rows // 2) + 0.5) * RESOLUTION_SCALE
    y = off_y - r - 0.5
    x = c - off_x + 0.5
    return float(x) / RESOLUTION_SCALE, float(y) / RESOLUTION_SCALE

def get_closest_valid_node(node, maze_map):
    rows, cols = maze_map.shape
    r, c = node
    if maze_map[r, c] == 0: return node
    
    # Spiral search for nearest empty cell
    for radius in range(1, 5): 
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if maze_map[nr, nc] == 0:
                        return (nr, nc)
    return node

def inflate_maze(original_map, scale):
    return np.kron(original_map, np.ones((scale, scale)))

def inflate_walls(maze_map, radius=1):
    """
    Adds a safety layer around walls.
    radius: How many extra layers of pixels to mark as occupied.
    """
    rows, cols = maze_map.shape
    new_map = maze_map.copy()
    
    # Find all current wall pixels
    wall_indices = np.argwhere(maze_map == 1)
    
    for r, c in wall_indices:
        # Mark neighbors as occupied
        r_min = max(0, r - radius)
        r_max = min(rows, r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(cols, c + radius + 1)
        
        new_map[r_min:r_max, c_min:c_max] = 1
        
    return new_map

# --- 2. A* ALGORITHM ---
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar_grid(maze_map, start, goal):
    rows, cols = maze_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze_map[neighbor] == 1: continue
                dist = np.sqrt(dr**2 + dc**2)
                tentative_g_score = g_score[current] + dist
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))
    return None

def pd_controller(current_pos, target_pos, velocity):
    kp, kd = 6.0, 5.0
    error = target_pos - current_pos
    force = (error * kp) - (velocity * kd)
    return np.clip(force, -1.0, 1.0)

# --- 3. LIVE VISUALIZATION FUNCTION ---
def update_plot(env, hires_map, start_node, goal_node, path_nodes, current_pos_node, ax_real, ax_grid):
    """
    Updates the existing axes with new data.
    """
    # 1. Real MuJoCo Image
    real_img = env.render()
    ax_real.clear()
    ax_real.imshow(real_img)
    ax_real.set_title("MuJoCo View (Live)")
    ax_real.axis('off')

    # 2. Grid Map View
    ax_grid.clear()
    ax_grid.imshow(hires_map, cmap='Greys', origin='upper')
    
    # Plot Static Elements (Path, Start, Goal)
    if path_nodes:
        rs = [p[0] for p in path_nodes]
        cs = [p[1] for p in path_nodes]
        ax_grid.plot(cs, rs, c='blue', linewidth=2, alpha=0.5, label="Planned Path")
        
    ax_grid.scatter(start_node[1], start_node[0], c='green', s=100, label="Start")
    ax_grid.scatter(goal_node[1], goal_node[0], c='red', s=150, marker='*', label="Goal")
    
    # Plot Dynamic Element (Current Agent Position)
    ax_grid.scatter(current_pos_node[1], current_pos_node[0], c='orange', s=120, edgecolors='black', label="Agent")
    
    ax_grid.set_title(f"Grid Tracking (Scale {RESOLUTION_SCALE}x)")
    ax_grid.legend(loc='upper right', fontsize='small')

    # Draw and Pause
    plt.draw()
    plt.pause(0.001)

def main_test():
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=1000, reward_type='dense')
    
    raw_map = np.array(env.unwrapped.maze.maze_map)
    original_shape = raw_map.shape
    hires_map = inflate_maze(raw_map, RESOLUTION_SCALE)
    safe_map = inflate_walls(hires_map, radius=2)
    
    # Initialize Plot
    if SHOW_VISUALIZATION:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    obs, _ = env.reset()
        
    start_xy = obs['observation'][:2]
    goal_xy = obs['desired_goal']
    
    # --- LOGIC: USE INTEGERS ---
    # We must use the integer version for A* and map checking
    start_node_int = world_to_grid(start_xy[0], start_xy[1], original_shape)
    goal_node_int = world_to_grid(goal_xy[0], goal_xy[1], original_shape)
    
    # Snap to safe integers
    start_node_int = get_closest_valid_node(start_node_int, safe_map)
    goal_node_int = get_closest_valid_node(goal_node_int, safe_map)
    
    # Plan Path using integers
    path_nodes = astar_grid(safe_map, start_node_int, goal_node_int)

    current_pos = obs['observation'][:2]
    
    # --- VISUALIZATION: USE FLOATS (OPTIONAL) ---
    if SHOW_VISUALIZATION:
        # 1. Calc float position purely for the "Yellow X" alignment check
        float_r, float_c = world_to_grid_float(current_pos[0], current_pos[1], original_shape)
        
        # 2. Update plot using the INTEGER nodes for the green/red dots
        # (You can pass float_r/float_c to your update_plot function if you modified it to show the X)
        update_plot(env, safe_map, start_node_int, goal_node_int, path_nodes, start_node_int, ax1, ax2)
        
        # Optional: Plot the Yellow X manually here if update_plot doesn't handle it
        ax2.scatter(float_c, float_r, c='gold', marker='x', s=150, linewidth=3, label="Exact Pos")
        ax2.legend()
        
        plt.ioff()
        plt.show() # This will block now
    
    print(f"Starting Data Generation...")

# --- 4. MAIN LOOP ---
def main():
    # Use rgb_array to capture frames for matplotlib
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=1000, reward_type='dense')
    
    raw_map = np.array(env.unwrapped.maze.maze_map)
    original_shape = raw_map.shape
    hires_map = inflate_maze(raw_map, RESOLUTION_SCALE)
    safe_map = inflate_walls(hires_map, radius=3)
    
    dataset = []
    success_count = 0
    
    # Initialize Plot only if visualization is on
    if SHOW_VISUALIZATION:
        plt.ion() # Interactive mode on
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    print(f"Starting Data Generation...")
    
    while success_count < TOTAL_EPISODES:
        obs, _ = env.reset()
        
        # Setup Start/Goal
        start_xy = obs['observation'][:2]
        goal_xy = obs['desired_goal']
        
        start_node = world_to_grid(start_xy[0], start_xy[1], original_shape)
        goal_node = world_to_grid(goal_xy[0], goal_xy[1], original_shape)
        
        start_node = get_closest_valid_node(start_node, safe_map)
        goal_node = get_closest_valid_node(goal_node, safe_map)
        
        # Plan Path
        path_nodes = astar_grid(safe_map, start_node, goal_node)
        
        if not path_nodes:
            print("Path planning failed. Retrying...")
            continue
            
        done = False
        current_path_idx = 0
        
        # 1. Find the closest node to our actual position
        start_world_pos = np.array(start_xy)
        closest_dist = float('inf')
        closest_idx = 0
        
        for i, node in enumerate(path_nodes):
            node_pos = np.array(grid_to_world(*node, original_shape))
            dist = np.linalg.norm(start_world_pos - node_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
                
        # 2. Set our initial target to be at least 'lookahead' steps ahead of that closest point
        # With Scale 10, lookahead=5 is about 0.5 meters, which is a good "Launch vector"
        current_path_idx = min(closest_idx + 2, len(path_nodes) - 1)
        episode_data = []
        step_count = 0
        
        print(f"\n--- Episode {success_count + 1} Started ---")
        
        while not done:
            step_count += 1
            
            current_pos = obs['observation'][:2]
            current_vel = obs['observation'][2:]
            
            # --- VISUALIZATION UPDATE ---
            if SHOW_VISUALIZATION:
                # Convert current pos to grid for plotting
                current_node = world_to_grid(current_pos[0], current_pos[1], original_shape)
                update_plot(env, safe_map, start_node, goal_node, path_nodes, current_node, ax1, ax2)
                
            
            # --- CONTROL LOGIC ---
            lookahead = 2 
            
            if current_path_idx < len(path_nodes):
                # Instead of aiming at 'current_path_idx', we aim ahead!
                target_idx = min(current_path_idx + lookahead, len(path_nodes) - 1)
                
                target_node = path_nodes[target_idx]
                target_world = np.array(grid_to_world(*target_node, original_shape))
                
                # Special Case: If we are nearing the end, aim exactly at the Goal
                if current_path_idx >= len(path_nodes) - 3:
                    target_world = goal_xy
            else:
                target_world = goal_xy

            action = pd_controller(current_pos, target_world, current_vel)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            if reward >= 0.85:
                terminated = True
            
            # --- PRINT INFO ---
            # Print Action (Forces) and Reward
            # Note: Reward in Sparse PointMaze is 0 (fail) or 1 (success).
            # If using Dense, it's negative distance.
            #print(f"Step: {step_count:03d} | Action: [{action[0]:.2f}, {action[1]:.2f}] | Reward: {reward:.4f}")
            
            episode_data.append({
                'obs': obs['observation'],
                'goal': obs['desired_goal'],
                'action': action,
                'reward': reward,
                'next_obs': next_obs['observation'],
                'done': terminated
            })
            
            obs = next_obs
            done = terminated or truncated
            
            # Waypoint Switching
            dist_threshold = 0.4 / RESOLUTION_SCALE 
            if np.linalg.norm(current_pos - target_world) < max(0.15, dist_threshold):
                current_path_idx += 1
        
        if terminated:
            dataset.extend(episode_data)
            success_count += 1
            print(f"--- Episode {success_count} Success! ---")
            print(f"Step: {step_count:03d} | Action: [{action[0]:.2f}, {action[1]:.2f}] | Reward: {reward:.4f}")
        else:
            print("--- Episode Failed (Time limit) ---")
            
    env.close()

    print(f"Success count: {success_count}")
    
    #Save Data
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} transitions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()