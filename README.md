# AI Pathfinder Simulation 

---

## **Project Overview**

**Advanced AI Pathfinder Simulation** is a comprehensive, interactive visualization tool built with **Pygame** that demonstrates multiple pathfinding algorithms on a dynamic grid. It supports weighted terrains, diagonal movement, real-time statistics, maze generation, and an analysis panel for comparing algorithm performance.

This project is ideal for learning, teaching, or exploring how classic and heuristic-based pathfinding algorithms behave under different conditions.

---

## **Features**

| Feature | Description |
|-------|-----------|
| **5 Pathfinding Algorithms** | A\*, Dijkstra, Greedy Best-First, BFS, DFS |
| **Weighted Terrains** | Normal (1), Grass (2), Water (5), Walls (∞) |
| **Diagonal Movement Toggle** | Enable/disable 8-directional movement |
| **Interactive Grid Editing** | Paint obstacles, terrain, start/end with mouse |
| **Random Obstacle Generation** | 5 density levels (10%–50%) |
| **Predefined Mazes** | Complex, Symmetric, Corridors (25×25 → scaled to 50×50) |
| **Live & Historical Stats** | Time, nodes explored, path length, space usage |
| **Algorithm Complexity Display** | Time & Space complexity shown per algorithm |
| **Analysis Panel** | Compare all algorithms side-by-side (live + last run) |
| **Smooth Visual Feedback** | Color-coded open/closed sets, path animation |
| **Customizable Brush Modes** | Wall, Normal, Grass, Water |

---

## **Screenshots**

*tbd*

---

## **Installation & Requirements**

### Prerequisites
- Python 3.6+
- Pygame

### Install Dependencies
```bash
pip install pygame
```

### Run the Project
```bash
python pathfinder.py
```

> Save the code as `pathfinder.py`

---

## **Controls**

| Key | Action |
|-----|--------|
| **LMB** | Place start → end → paint with current brush |
| **RMB** | Erase (reset node) |
| **SPACE** | Run selected algorithm |
| **C** | Clear entire board |
| **R** | Generate random obstacles (current density) |
| **M / , / .** | Load predefined mazes: Complex / Symmetric / Corridors |
| **D** | Toggle diagonal movement |
| **A** | Toggle analysis panel |
| **1–5** | Set obstacle density: 10%, 20%, 30%, 40%, 50% |
| **6–0** | Select algorithm: A\*, Dijkstra, Greedy, BFS, DFS |
| **O / N / G / W** | Brush: Wall / Normal / Grass / Water |

---

## **Algorithms Implemented**

| Algorithm | Heuristic | Optimal | Time | Space |
|---------|----------|--------|------|-------|
| **A\* Search** | Manhattan | Yes | `O(b^d)` | `O(b^d)` |
| **Dijkstra** | None | Yes | `O(V + E log V)` | `O(V)` |
| **Greedy BFS** | Manhattan | No | `O(b^d)` | `O(b^d)` |
| **BFS** | None | Yes | `O(V + E)` | `O(V)` |
| **DFS** | None | No | `O(V + E)` | `O(V)` |

> `b` = branching factor, `d` = depth of solution, `V` = vertices, `E` = edges

---

## **Grid & Visuals**

- **Grid Size**: 50×50 (800×800 px)
- **Cell Size**: 16×16 px
- **UI Panel**: 300 px wide (right side)
- **Analysis Panel**: 160 px tall (bottom, toggleable)

### Color Legend

| Color | Meaning |
|------|--------|
| **Blue** | Start node |
| **Yellow** | End node |
| **Purple** | Final path |
| **Green** | Open set (being explored) |
| **Red** | Closed set (already explored) |
| **Gray** | Obstacle / Wall |
| **Light Green** | Grass (cost 2) |
| **Light Blue** | Water (cost 5) |
| **White** | Normal tile (cost 1) |

---

## **Code Structure**

```plaintext
├── Configuration
│   ├── Screen, Grid, Colors, Fonts
│   └── Constants (density, brushes, mazes)
│
├── Node Class
│   ├── Properties: pos, color, weight, neighbors
│   └── Methods: make_start(), make_path(), draw(), etc.
│
├── Algorithms
│   ├── generic_search() → A*, Dijkstra, Greedy
│   ├── BFS, DFS (separate)
│   └── reconstruct_path()
│
├── Maze & Obstacles
│   ├── Random generation
│   └── Predefined scaled mazes
│
├── UI & Drawing
│   ├── draw_ui(), draw_analysis_panel()
│   └── Real-time stats tracking
│
└── Main Loop
    ├── Event handling
    └── State management
```

---

## **How It Works**

1. **Grid Initialization**  
   → 50×50 `Node` objects created.

2. **User Interaction**  
   → Click to place start/end, paint terrain.

3. **Run Algorithm**  
   → `update_neighbors()` called with diagonal flag.  
   → Selected algorithm runs with live visualization.

4. **Path Reconstruction**  
   → Backtrack from `end.previous` → draw smooth lines.

5. **Stats Collection**  
   → Time, nodes explored, max space used, path length.

6. **Analysis Panel**  
   → Shows live stats during run, last run after completion.

---

## **Enjoy Pathfinding!**

> Press **SPACE** to watch AI find the optimal path in real time!


*Made with ❤️ for algorithm enthusiasts*  
*Last Updated: November 10, 2025*
