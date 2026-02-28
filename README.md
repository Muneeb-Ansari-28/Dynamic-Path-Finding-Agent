# Dynamic Pathfind Agent

A Pygame-based interactive pathfinding visualizer implementing **GBFS** and **A\*** with Manhattan and Euclidean heuristics, plus dynamic obstacle spawning and real-time re-planning.

---

## Features

| Feature | Details |
|---|---|
| **Algorithms** | Greedy Best-First Search (GBFS), A* Search |
| **Heuristics** | Manhattan Distance, Euclidean Distance |
| **Grid** | Configurable rows × cols (5–40 × 5–60) |
| **Map Editor** | Click to place/remove walls; move Start & Goal |
| **Random Map** | Adjustable obstacle density (5–70 %) |
| **Dynamic Mode** | Obstacles spawn mid-run; auto replanning |
| **Visualization** | Frontier (yellow), Visited (blue), Path (green), Agent (pink) |
| **Metrics** | Nodes visited, path cost, execution time (ms) |

---

## Requirements

```
Python 3.8+
pygame
```

## Installation

```bash
pip install pygame
```

## Run

```bash
python pathfinding_agent.py
```

---

## Controls

| Control | Action |
|---|---|
| **R** | Run Search |
| **G** | Generate random map |
| **C** | Clear grid |
| **Space** | Animate agent |
| **Left-click grid** | Place / remove wall (or move Start/Goal) |

---

## UI Guide

1. **Select Algorithm**: A* or GBFS button (top-right panel)
2. **Select Heuristic**: Manhattan or Euclidean
3. **Edit Mode**: Wall / Start / Goal — then click on grid
4. **Adjust sliders**: Rows, Cols, Density %, Anim Speed, Spawn Prob %
5. **Generate Map**: Creates a random maze at selected density
6. **▶ RUN SEARCH**: Animates the exploration + path
7. **▶ Animate Agent**: Walks the agent along the found path
8. **⚡ Dynamic Mode**: Toggle random obstacle spawning during agent movement

---

## Algorithm Details

### Greedy Best-First Search (GBFS)
- **f(n) = h(n)**
- Only considers the heuristic — no path cost
- Fast but **not guaranteed optimal**

### A* Search
- **f(n) = g(n) + h(n)**
- Balances path cost and heuristic
- **Optimal** with admissible heuristic (Manhattan on 4-connected grid)

### Heuristics
- **Manhattan**: `|x1−x2| + |y1−y2|` — ideal for 4-connected grids
- **Euclidean**: `√((x1−x2)² + (y1−y2)²)` — can overestimate on grids; leads to more exploration

---

## Dynamic Re-planning

When **Dynamic Mode** is ON:
- Each frame, empty cells have a small probability (`Spawn Prob %`) of becoming walls
- If a new wall lands on the agent's current planned path, the agent immediately replans from its current position using the selected algorithm
- If no wall intersects the path, no recomputation is done (efficiency optimization)
- The re-plan counter tracks how many times replanning occurred
