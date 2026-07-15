import heapq
import numpy as np
import math

def heuristic(a, b, metric="Manhattan"):
    y1, x1 = a
    y2, x2 = b
    
    if metric == "Euclidean":
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elif metric == "Chebyshev":
        return max(abs(x1 - x2), abs(y1 - y2))
    else:
        return abs(x1 - x2) + abs(y1 - y2)

def solve_a_star(grid, heuristic_metric="Manhattan"):
    start = tuple(np.argwhere(grid == 2)[0])
    goal = tuple(np.argwhere(grid == 3)[0])
    
    pq = [(0, 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_nodes = [] 
    
    while pq:
        _, current_cost, current = heapq.heappop(pq)
        visited_nodes.append(current)
        
        if current == goal:
            break
            
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Wall check
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue # If it's a wall
                    
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal, metric=heuristic_metric)
                    heapq.heappush(pq, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
                    
    # Final path
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    return path[::-1], visited_nodes

class QLearningAgent:
    def __init__(self, grid, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.grid = grid
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.height, self.width = grid.shape
        
        self.actions = [0, 1, 2, 3]
        self.action_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Q table setup
        self.q_table = {}
        for r in range(self.height):
            for c in range(self.width):
                if grid[r, c] != 1:
                    self.q_table[(r, c)] = np.zeros(4)
                    
        self.start = tuple(np.argwhere(grid == 2)[0])
        self.goal = tuple(np.argwhere(grid == 3)[0])

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions) 
        else:
            return np.argmax(self.q_table[state]) 

    def step(self, state, action):
        dr, dc = self.action_vectors[action]
        next_state = (state[0] + dr, state[1] + dc)
        
        # Wall check
        if (next_state[0] < 0 or next_state[0] >= self.height or 
            next_state[1] < 0 or next_state[1] >= self.width or 
            self.grid[next_state[0], next_state[1]] == 1):
            return state, -5, False # penalty
            
        if next_state == self.goal:
            return next_state, 100, True # reward
            
        return next_state, -1, False

    def train_episode(self):
        state = self.start
        done = False
        steps = 0
        max_steps = self.height * self.width * 2
        
        while not done and steps < max_steps:
            action = self.choose_action(state)
            next_state, reward, done = self.step(state, action)
            
            # Bellman Equation Update
            old_value = self.q_table[state][action]
            next_max = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
            
            self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            
            state = next_state
            steps += 1
            
        return steps

    def get_optimal_path(self):
        state = self.start
        path = [state]
        visited = set([state])
        
        while state != self.goal:
            action = np.argmax(self.q_table[state])
            dr, dc = self.action_vectors[action]
            next_state = (state[0] + dr, state[1] + dc)
    
            if next_state in visited or next_state not in self.q_table:
                return None
                
            path.append(next_state)
            visited.add(next_state)
            state = next_state
            
        return path