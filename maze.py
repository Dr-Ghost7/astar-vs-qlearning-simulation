import numpy as np
import random

def generate_maze(width, height, loop_rate):
    w = (width // 2) * 2 + 1
    h = (height // 2) * 2 + 1
    
    grid = np.ones((h, w), dtype=int)
    stack = [(1, 1)]
    grid[1, 1] = 0
    
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < h - 1 and 0 < ny < w - 1 and grid[nx, ny] == 1:
                neighbors.append((nx, ny, dx, dy))
                
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            grid[cx + dx//2, cy + dy//2] = 0
            grid[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
            
    if loop_rate > 0:
        walls = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if grid[r, c] == 1:
                    walls.append((r, c))
        
        num_to_remove = int(len(walls) * loop_rate)
        walls_to_break = random.sample(walls, min(num_to_remove, len(walls)))
        for r, c in walls_to_break:
            grid[r, c] = 0

    grid[1, 1] = 2
    grid[h - 2, w - 2] = 3
    return grid