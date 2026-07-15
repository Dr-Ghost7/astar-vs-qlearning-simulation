import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from maze import generate_maze
from solvers import solve_a_star, QLearningAgent

st.set_page_config(page_title="Maze Path Finding Simulation", layout="wide")
st.title("A* Search vs Q-Learning Pathfinding Simulation")
# Guide
st.markdown("""
### Quick Start Guide
1. **Setup:** In the sidebar controls, you can change the maze size, complexity, and colors.
2. **A\* Search:** Press **Run A\* Solver** and see the math-based GPS instantly calculate the absolute shortest path.
3. **Q-Learning:** Observe **Watch AI explore** to see the training in action, then press **Train & Run RL Agent** to have the AI learn the maze.
4. **Compare:** Scroll down and click **Run Benchmark** to view their performance data side-by-side.
""")
st.write("---")
# Sidebar Configuration Controls
st.sidebar.header("Simulation Settings")
grid_size = st.sidebar.slider(
    "Maze Grid Size", 
    min_value=11, max_value=31, value=15, step=2
)

loop_rate = st.sidebar.slider(
    "Maze Complexity (Loop Rate)", 
    min_value=0.0, max_value=0.3, value=0.1, step=0.05,
    help="Calculates ratio of dead-ends converted to loops. 0.0 means a 'perfect' maze with only one path to the solution. Higher values provide multiple alternatives."
)

animation_speed = st.sidebar.slider(
    "Animation Delay (s)", 
    min_value=0.005, max_value=0.2, value=0.02, step=0.005
)

st.sidebar.header("RL Hyperparameters")
episodes = st.sidebar.slider(
    "Training Episodes", 
    min_value=50, max_value=1200, value=400, step=50,
    help="The total number of full game attempts (from entrance to exit, or until running out of steps) the AI gets to build its memory table."
)

epsilon_rate = st.sidebar.slider(
    "Exploration Rate (Epsilon)", 
    min_value=0.1, max_value=0.9, value=0.4, step=0.05,
    help="The chance (0.40 = 40%) that the AI takes a totally random step to explore new areas, vs 60% chance to trust it’s memory to take the best known step."
)

max_steps_multiplier = st.sidebar.slider(
    "Max Steps Multiplier", 
    min_value=1, max_value=5, value=2, step=1,
    help="Sets the per-run step ceiling to be: (Grid Size x Grid Size) x Multiplier. This is a safety limit that will make a reset happen if the AI gets stuck pacing in circles."
)
st.sidebar.header("A* Algorithm Tweaks")
heuristic_type = st.sidebar.selectbox(
    "A* Distance Heuristic",
    options=["Manhattan", "Euclidean", "Chebyshev"],
    help="Manhattan: Grid-based vertical/horizontal steps. Euclidean: Absolute straight-line distance. Chebyshev: Allows diagonal calculation weight."
)
computed_max_steps = grid_size * grid_size * max_steps_multiplier

st.sidebar.header("Interface Customization")
user_path_color = st.sidebar.color_picker("Final Path Route Color", value="#18ECF3")
user_search_color = st.sidebar.color_picker("Exploration Tracker Color", value="#1A5961")

# Maze gen
if ('current_size' not in st.session_state or 
    st.session_state.current_size != grid_size or 
    'current_loop' not in st.session_state or 
    st.session_state.current_loop != loop_rate or 
    st.sidebar.button("Generate New Maze")):
    
    st.session_state.maze_grid = generate_maze(grid_size, grid_size, loop_rate)
    st.session_state.current_size = grid_size
    st.session_state.current_loop = loop_rate
    st.session_state.a_star_final_grid = None
    st.session_state.rl_final_grid = None

# Base Visual Palette
HEX_COLORS = ["#F0F2F6", "#1E1E24", "#007FFF", "#FF4B4B", user_search_color, user_path_color]
CUSTOM_CMAP = ListedColormap(HEX_COLORS)

def render_maze_with_agent(grid_matrix, agent_pos=None, agent_color="#FFD700"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(grid_matrix, cmap=CUSTOM_CMAP, vmin=0, vmax=5)
    
    if agent_pos is not None:
        ax.plot(agent_pos[1], agent_pos[0], marker='o', color=agent_color, 
                markersize=14, markeredgecolor='white', markeredgewidth=2, 
                zorder=5)
        
    ax.axis("off")
    fig.patch.set_facecolor('#0E1117')
    return fig

col1, spacer_col, col2 = st.columns([1, 0.15, 1])

# Col 1
with col1:
    st.subheader("Classical A* Search Tracker")
    a_star_placeholder = st.empty()
    
    if st.session_state.a_star_final_grid is not None:
        a_star_placeholder.pyplot(render_maze_with_agent(st.session_state.a_star_final_grid))
    else:
        a_star_placeholder.pyplot(render_maze_with_agent(st.session_state.maze_grid))
    
    btn_col_a1, _, btn_col_a2 = st.columns([1, 2.7, 1])
    with btn_col_a1:
        run_a_star = st.button("Run A* Solver")
    with btn_col_a2:
        if st.button("Clear A* Path"):
            st.session_state.a_star_final_grid = None
            st.rerun()
            
    if run_a_star:
        st.session_state.a_star_final_grid = None
        a_star_placeholder.pyplot(render_maze_with_agent(st.session_state.maze_grid))
        
        path, visited = solve_a_star(st.session_state.maze_grid, heuristic_metric=heuristic_type)
        animated_grid = st.session_state.maze_grid.copy()

        for node in visited:
            if animated_grid[node[0], node[1]] not in [2, 3]:
                animated_grid[node[0], node[1]] = 4 
                a_star_placeholder.pyplot(render_maze_with_agent(animated_grid, agent_pos=node, agent_color=user_search_color))
                time.sleep(animation_speed)
                
        for node in path:
            if animated_grid[node[0], node[1]] not in [2, 3]:
                animated_grid[node[0], node[1]] = 5 
            a_star_placeholder.pyplot(render_maze_with_agent(animated_grid, agent_pos=node, agent_color=user_path_color))
            time.sleep(animation_speed)
        
        st.session_state.a_star_final_grid = animated_grid

    st.markdown("""
    **How A\* Search Works:** Consider A\* to be similar to a GPS mapping application that knows where to go. Unlike the other strategies, it does not assume anything. From the beginning itself, A\* uses mathematical calculations to find out how far is the red exit (destination) from its current location. It considers all available paths and picks up the one which brings it nearer to the destination.
    """)

    with st.expander("A* Cost Formula"):
        st.markdown(R"""
        ### Distance Equation
        A\* prioritizes search directions using this total cost formula:
        $$f(n) = g(n) + h(n)$$
        
        * **$f(n)$**: Total estimated cost of path route.
        * **$g(n)$**: Actual steps taken from start node to current node.
        * **$h(n)$**: Estimated distance to red exit via **Manhattan Distance**:
        $$h(n) = |x_{current} - x_{goal}| + |y_{current} - y_{goal}|$$
        
        **Mathematical Example:**
        If the tracker has traveled $6$ steps ($g = 6$) and the goal coordinates are physically $4$ rows and $2$ columns away ($h = 4 + 2 = 6$):
        $$f(n) = 6 + 6 = 12$$
        """)
# Col 2
with col2:
    st.subheader("Q-Learning Agent")
    rl_placeholder = st.empty()
    
    if st.session_state.rl_final_grid is not None:
        rl_placeholder.pyplot(render_maze_with_agent(st.session_state.rl_final_grid))
    else:
        rl_placeholder.pyplot(render_maze_with_agent(st.session_state.maze_grid))
        
    watch_training = st.checkbox("Watch AI explore during training (Slows it down)", value=False)
    st.caption(f"Active Step Limit: **{computed_max_steps}** moves allowed per episode.")

    st.write("") 
    
    btn_col_r1, _, btn_col_r2 = st.columns([1.5, 2.7, 1])
    with btn_col_r1:
        run_rl = st.button("Train & Run RL Agent")
    with btn_col_r2:
        if st.button("Clear RL Path"):
            st.session_state.rl_final_grid = None
            st.rerun()

    st.write("")
    st.write("### Live Q-Value Matrix")
    
    q_table_placeholder = st.empty()
    q_table_placeholder.info("Start training to see the Q-table update.")
    
    st.write("")
            
    if run_rl:
        st.session_state.rl_final_grid = None
        rl_placeholder.pyplot(render_maze_with_agent(st.session_state.maze_grid))
        
        agent = QLearningAgent(st.session_state.maze_grid, epsilon=epsilon_rate)
        status_text = st.empty()
        
        for ep in range(episodes):
            state = agent.start
            done = False
            steps = 0
            
            while not done and steps < computed_max_steps:
                action = agent.choose_action(state)
                next_state, reward, done = agent.step(state, action)
                
                old_value = agent.q_table[state][action]
                next_max = np.max(agent.q_table[next_state]) if next_state in agent.q_table else 0
                agent.q_table[state][action] = old_value + agent.alpha * (reward + agent.gamma * next_max - old_value)
                
                
                if watch_training and ep % 10 == 0:
                    temp_grid = st.session_state.maze_grid.copy()
                    if next_state not in [agent.start, agent.goal]:
                        temp_grid[next_state[0], next_state[1]] = 4 
                    rl_placeholder.pyplot(render_maze_with_agent(temp_grid, agent_pos=next_state, agent_color="#FFD700"))
                    
                    
                    current_q = agent.q_table.get(next_state, np.zeros(4))
                    df_q = pd.DataFrame([current_q], columns=["Up", "Down", "Left", "Right"], index=[f"Coordinate {next_state}"])
                    q_table_placeholder.dataframe(df_q.style.format("{:.2f}"))
                    time.sleep(0.001)
                
                state = next_state
                steps += 1
                
            if ep % 50 == 0:
                status_text.text(f"Training Progress: Episode {ep}/{episodes}...")
        
        status_text.text("Training Complete! Tracing learned strategy path...")
        
        
        rl_path = agent.get_optimal_path()
        animated_rl_grid = st.session_state.maze_grid.copy()
        
        for node in rl_path:
            if animated_rl_grid[node[0], node[1]] not in [2, 3]:
                animated_rl_grid[node[0], node[1]] = 5 
            rl_placeholder.pyplot(render_maze_with_agent(animated_rl_grid, agent_pos=node, agent_color=user_path_color))
            
           
            current_q = agent.q_table.get(node, np.zeros(4))
            df_q = pd.DataFrame([current_q], columns=["Up", "Down", "Left", "Right"], index=[f"Coordinate {node}"])
            q_table_placeholder.dataframe(df_q.style.format("{:.2f}"))
            time.sleep(animation_speed)
        
        st.session_state.rl_final_grid = animated_rl_grid

    st.markdown("""
    **How Q-Learning Works:** Imagine an RL agent as a puppy learning to perform a new trick. The puppy has no maps, math abilities, or knowledge about the location of the exit. During training, it moves randomly bumping into walls. Every time it bumps into a wall, it receives a little penalty. But when it accidentally discovers the red exit, it receives a huge reward. After hundreds of such attempts, the puppy remembers those paths along which it received rewards. And even though the path may be strange and long, it does not matter for the puppy since it was discovered first.
    """)


    with st.expander("Q-Learning Bellman Formula"):
        st.markdown(R"""
        ### The Bellman Equation
        Memory matrix adjustments calculated with this equation:
        $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
        
        * **$Q(s, a)$**: Value weight of current move state.
        * **$\alpha$ (Learning Rate)**: Memory update speed weight factor.
        * **$R$**: Reward values ($-5$ wall, $-1$ step, $+100$ goal).
        * **$\gamma$**: Discount factor weight for future moves evaluation.
        * **$\max_{a'} Q(s', a')$**: Maximized memory value of potential next steps.
        """)

st.write("---")
st.header("Performance Benchmarking")
st.markdown("""
This benchmark tests the efficiency of paths that these two algorithms generate using 500 trials conducted on random mazes.
The benchmark checks how many steps the algorithms take in order to reach the end point in comparison to mathematical optimizations and reinforcement learning.
""")

if st.button("Run Benchmark"):
    progress_bar = st.progress(0)
    status_msg = st.empty()
    
    a_star_steps_log = []
    rl_steps_log = []
    
    status_msg.text("Running A* Search trials...")
    for i in range(250):
        test_grid = generate_maze(grid_size, grid_size, loop_rate)
        path, _ = solve_a_star(test_grid)
        a_star_steps_log.append(len(path))
        progress_bar.progress(int((i + 1) / 500 * 100))
        
    status_msg.text("Running Q-Learning training trials...")
    test_agent = QLearningAgent(st.session_state.maze_grid, epsilon=epsilon_rate)
    
    for ep in range(250):
        steps_taken = test_agent.train_episode()
        rl_steps_log.append(steps_taken)
        progress_bar.progress(int((250 + ep + 1) / 500 * 100))
        
    status_msg.text("Processing benchmark data...")
    progress_bar.empty()
    status_msg.empty()
    
    import seaborn as sns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#0E1117')
    
    # Bell Curves
    ax1.set_facecolor('#1E1E24')
    sns.kdeplot(a_star_steps_log, fill=True, color="#007FFF", label="A* Search", ax=ax1, bw_adjust=1.5)
    sns.kdeplot(rl_steps_log, fill=True, color=user_path_color, label="Q-Learning", ax=ax1, bw_adjust=0.8)
    ax1.set_title("Path Length Distribution (Bell Curve)", color="white", fontsize=12)
    ax1.set_xlabel("Steps to Goal", color="white")
    ax1.set_ylabel("Probability Density", color="white")
    ax1.tick_params(colors="white")
    ax1.legend()
    ax1.grid(True, alpha=0.1)
    
    # The Learning Curve
    ax2.set_facecolor('#1E1E24')
    ax2.plot(range(len(rl_steps_log)), rl_steps_log, color=user_path_color, alpha=0.6, linewidth=1.5)
    ax2.set_title("Q-Learning Convergence Rate", color="white", fontsize=12)
    ax2.set_xlabel("Training Episodes", color="white")
    ax2.set_ylabel("Steps Taken", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout(w_pad=4.0)
    st.pyplot(fig)
    
    st.markdown(f"""
    ### Benchmark Analysis
    
    * **Distribution Profile (Left Graph):** A* Search follows a spike distribution that is both very thin and densely populated, owing to the fact that it is highly deterministic, which means that it is able to always calculate the mathematically optimal path. Q-Learning follows a relatively wide distribution.
    * **Convergence Profile (Right Graph):** The learning curve indicates how the Q-learning algorithm optimizes its policy. During the first few episodes of learning, there is a large number of steps since the agent is exploring randomly; but thereafter, a decay in the steps occurs consistently.
    """)

if run_a_star:
        st.session_state.a_star_final_grid = None
        
        path, visited = solve_a_star(st.session_state.maze_grid, heuristic_metric=heuristic_type)
        animated_grid = st.session_state.maze_grid.copy()
        
        # Use a text status bar to show progress instantly without lag
        status_bar = st.progress(0)
        
        # 1. Fast track exploration tracking
        for i, node in enumerate(visited):
            if animated_grid[node[0], node[1]] not in [2, 3]:
                animated_grid[node[0], node[1]] = 4 
            
            # Update a progress bar instead of hammering the server with plots
            status_bar.progress(min((i + 1) / len(visited), 1.0))
            time.sleep(animation_speed)
                
        # 2. Trace final path
        for node in path:
            if animated_grid[node[0], node[1]] not in [2, 3]:
                animated_grid[node[0], node[1]] = 5 
        
        # Clear the status bar and render the final map ONCE
        status_bar.empty()
        st.session_state.a_star_final_grid = animated_grid
        a_star_placeholder.pyplot(render_maze_with_agent(st.session_state.a_star_final_grid))    