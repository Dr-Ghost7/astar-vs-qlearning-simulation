# A* vs Q-Learning Simulation

A comparison of a classical search algorithm (A*) against a reinforcement learning algorithm (Q-Learning) on randomly generated mazes, evaluating path optimality and how many episodes Q-Learning needs before it converges to a working solution.

## What this does

- Generates maze environments (`maze.py`) and solves them two ways:
  - **A\*** — a deterministic pathfinding search that always finds the shortest path.
  - **Q-Learning** — a reinforcement learning agent (`solvers.py`) that learns a policy through trial-and-error over a fixed training budget (150 episodes), with no prior knowledge of the maze structure.
- Runs both solvers across a batch of mazes and compares path length and convergence.
- Serves results through a small app (`app.py`), including a scatter plot comparing path lengths maze-by-maze.

## Benchmark results

Run across 250 randomly generated mazes, with Q-Learning capped at 150 training episodes per maze:

- **Convergence**: Q-Learning found a valid solution within the training budget on **79 of 250 mazes (31.6%)**. The remaining mazes weren't solved within 150 episodes — raising the episode cap increases this rate (worth exploring further).
- **Path quality**: On the mazes it did solve, Q-Learning's paths averaged **0.1 extra steps** compared to A*'s optimal route. In the scatter plot, points on the red dashed line matched A*'s path length exactly; points above it show how much longer that particular path was.

**Takeaway**: When Q-Learning converges within a limited episode budget, it tends to find paths very close to optimal — but under a tight budget, it fails to converge at all on roughly two-thirds of mazes. This highlights the classic trade-off between A*'s guaranteed optimality/speed on known environments and Q-Learning's ability to solve problems with no upfront map knowledge, at the cost of needing many training episodes.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

## Possible next steps

- Measure and compare wall-clock solve time between A* and Q-Learning.
- Test how the 31.6% convergence rate changes as the episode budget is raised (e.g. 300, 500 episodes).
- Try alternate reward shaping to speed up Q-Learning convergence.
