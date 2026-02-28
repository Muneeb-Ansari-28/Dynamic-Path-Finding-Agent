"""
Dynamic Pathfinding Agent
=========================
Implements GBFS and A* with Manhattan/Euclidean heuristics on a Pygame grid.
Supports dynamic obstacle spawning and real-time re-planning.

Requirements:
    pip install pygame

Run:
    python pathfinding_agent.py
"""

import pygame
import math
import heapq
import random
import time
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & THEME
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 820
PANEL_W = 320
GRID_W = SCREEN_W - PANEL_W   # 960 px for the grid area

FPS = 60

# Cell type identifiers
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
VISITED  = 4
FRONTIER = 5
PATH     = 6
AGENT    = 7

# Color palette  (dark-mode, cyberpunk aesthetic)
BG          = (10,  12,  20)
PANEL_BG    = (16,  19,  32)
ACCENT      = (0,  200, 255)
ACCENT2     = (255, 80, 160)

C_EMPTY    = (22,  26,  42)
C_WALL     = (45,  52,  80)
C_START    = (0,  200, 255)
C_GOAL     = (255, 180,  40)
C_VISITED  = (30,  55, 110)
C_FRONTIER = (250, 220,  50)
C_PATH     = (40,  210,  90)
C_AGENT    = (255,  80, 160)
C_GRID     = (30,  35,  55)
C_TEXT     = (220, 230, 255)
C_SUBTEXT  = (100, 115, 155)
C_BUTTON   = (30,  38,  70)
C_BTN_HOV  = (45,  58, 100)
C_BTN_ACT  = (0,  160, 210)
C_SUCCESS  = (40,  210,  90)
C_ERROR    = (220,  60,  60)
C_WARNING  = (255, 160,  40)

pygame.font.init()
FONT_TITLE  = pygame.font.SysFont("Consolas", 20, bold=True)
FONT_LABEL  = pygame.font.SysFont("Consolas", 14, bold=True)
FONT_SMALL  = pygame.font.SysFont("Consolas", 12)
FONT_METRIC = pygame.font.SysFont("Consolas", 26, bold=True)
FONT_TINY   = pygame.font.SysFont("Consolas", 11)

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY QUEUE
# ─────────────────────────────────────────────────────────────────────────────
class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0   # tie-breaker

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        _, _, item = heapq.heappop(self._heap)
        return item

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def to_set(self):
        return {item for _, _, item in self._heap}


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTICS
# ─────────────────────────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────
def get_neighbors(grid, row, col):
    """4-connected neighbors that are not walls."""
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    result = []
    for dr, dc in dirs:
        r, c = row+dr, col+dc
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] != WALL:
            result.append((r, c))
    return result


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def gbfs(grid, start, goal, heuristic_fn):
    """
    Greedy Best-First Search  f(n) = h(n)
    Returns: (path, visited_order, frontier_snapshots)
    """
    h = heuristic_fn
    open_set = PriorityQueue()
    open_set.push(h(start, goal), start)
    came_from = {}
    visited = []
    open_nodes = {start}
    closed = set()

    while open_set:
        current = open_set.pop()
        open_nodes.discard(current)

        if current == goal:
            return reconstruct_path(came_from, current), visited, True

        if current in closed:
            continue
        closed.add(current)
        visited.append(current)

        for nb in get_neighbors(grid, *current):
            if nb not in closed and nb not in open_nodes:
                came_from[nb] = current
                open_set.push(h(nb, goal), nb)
                open_nodes.add(nb)

    return [], visited, False


def astar(grid, start, goal, heuristic_fn):
    """
    A* Search  f(n) = g(n) + h(n)
    Returns: (path, visited_order, success)
    """
    h = heuristic_fn
    open_set = PriorityQueue()
    open_set.push(h(start, goal), start)
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    visited = []
    open_nodes = {start}
    closed = set()

    while open_set:
        current = open_set.pop()
        open_nodes.discard(current)

        if current == goal:
            return reconstruct_path(came_from, current), visited, True

        if current in closed:
            continue
        closed.add(current)
        visited.append(current)

        for nb in get_neighbors(grid, *current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + h(nb, goal)
                if nb not in closed:
                    open_set.push(f, nb)
                    open_nodes.add(nb)

    return [], visited, False


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_text(surface, text, font, color, x, y, center=False):
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)
    surface.blit(surf, rect)
    return rect


class Button:
    def __init__(self, rect, label, active=False, color_active=C_BTN_ACT,
                 color_normal=C_BUTTON, toggle=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.active = active
        self.color_active = color_active
        self.color_normal = color_normal
        self.toggle = toggle
        self.hovered = False

    def draw(self, surface):
        color = self.color_active if self.active else (C_BTN_HOV if self.hovered else self.color_normal)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, ACCENT if self.active else C_GRID, self.rect, 1, border_radius=6)
        draw_text(surface, self.label, FONT_LABEL,
                  C_TEXT if self.active else C_SUBTEXT,
                  self.rect.centerx, self.rect.centery, center=True)

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if self.toggle:
                self.active = not self.active
            return True
        return False


class Slider:
    def __init__(self, rect, label, min_val, max_val, value, integer=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.integer = integer
        self.dragging = False

    @property
    def display_value(self):
        return int(self.value) if self.integer else round(self.value, 2)

    def _val_from_x(self, x):
        ratio = (x - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))
        v = self.min_val + ratio * (self.max_val - self.min_val)
        return int(round(v)) if self.integer else v

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._val_from_x(event.pos[0])

    def draw(self, surface):
        draw_text(surface, f"{self.label}: {self.display_value}",
                  FONT_LABEL, C_TEXT, self.rect.x, self.rect.y - 18)
        pygame.draw.rect(surface, C_BUTTON, self.rect, border_radius=4)
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = int(ratio * self.rect.width)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
        pygame.draw.rect(surface, ACCENT, fill_rect, border_radius=4)
        handle_x = self.rect.x + fill_w
        pygame.draw.circle(surface, C_TEXT, (handle_x, self.rect.centery), 8)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class PathfindingApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent  |  GBFS & A*")
        self.clock = pygame.time.Clock()

        # Grid config
        self.rows = 20
        self.cols = 30
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

        # Algorithm state
        self.algorithm  = "A*"      # "A*" | "GBFS"
        self.heuristic  = "Manhattan"  # "Manhattan" | "Euclidean"
        self.edit_mode  = "Wall"    # "Wall" | "Start" | "Goal"
        self.mouse_down = False

        # Visualization state
        self.visited_cells   = []
        self.frontier_cells  = set()
        self.path_cells      = []
        self.agent_pos       = None
        self.agent_path      = []
        self.agent_step      = 0

        # Animation
        self.animating      = False
        self.visit_index    = 0
        self.show_path      = False
        self.anim_speed     = 20   # cells revealed per frame during exploration
        self.agent_running  = False
        self.agent_timer    = 0
        self.agent_interval = 80   # ms between agent steps

        # Dynamic mode
        self.dynamic_mode   = False
        self.spawn_prob     = 0.05
        self.replanning     = False
        self.replan_count   = 0

        # Metrics
        self.nodes_visited  = 0
        self.path_cost      = 0
        self.exec_time_ms   = 0
        self.status_msg     = "Ready"
        self.status_color   = C_SUBTEXT

        # Obstacle density (slider value)
        self.density        = 0.30

        self._build_ui()

    # ── UI Layout ──────────────────────────────────────────────────────────
    def _build_ui(self):
        px = GRID_W + 14   # panel x start
        pw = PANEL_W - 28  # panel content width

        # Algorithm buttons
        self.btn_astar = Button((px, 60, pw//2-4, 34), "A*",  active=True, toggle=False)
        self.btn_gbfs  = Button((px+pw//2+4, 60, pw//2-4, 34), "GBFS", toggle=False)

        # Heuristic buttons
        self.btn_manh = Button((px, 128, pw//2-4, 34), "Manhattan", active=True, toggle=False)
        self.btn_eucl = Button((px+pw//2+4, 128, pw//2-4, 34), "Euclidean", toggle=False)

        # Edit mode buttons
        self.btn_wall  = Button((px,           196, pw//3-3, 30), "Wall",  active=True, toggle=False)
        self.btn_sedit = Button((px+pw//3+3,   196, pw//3-3, 30), "Start", toggle=False)
        self.btn_gedit = Button((px+2*pw//3+6, 196, pw//3-3, 30), "Goal",  toggle=False)

        # Sliders
        self.sl_rows    = Slider((px, 272, pw, 10), "Rows",    5, 40, self.rows)
        self.sl_cols    = Slider((px, 318, pw, 10), "Cols",   5, 60, self.cols)
        self.sl_density = Slider((px, 364, pw, 10), "Density %", 5, 70,
                                  int(self.density*100), integer=True)
        self.sl_speed   = Slider((px, 410, pw, 10), "Anim Speed", 1, 60, self.anim_speed)
        self.sl_dynprob = Slider((px, 456, pw, 10), "Spawn Prob %", 1, 20,
                                  int(self.spawn_prob*100), integer=True)

        # Action buttons
        self.btn_generate = Button((px,           502, pw//2-4, 36), "Generate Map",  color_active=C_BTN_ACT)
        self.btn_clear    = Button((px+pw//2+4,   502, pw//2-4, 36), "Clear Grid",    color_active=C_BTN_ACT)
        self.btn_run      = Button((px,           548, pw,      40), "▶  RUN SEARCH", color_active=(20,160,80))
        self.btn_dynamic  = Button((px,           600, pw,      36), "⚡ Dynamic Mode OFF",
                                    toggle=True, color_active=(180, 60, 0))
        self.btn_animate  = Button((px,           646, pw,      36), "▶ Animate Agent",
                                    color_active=(60, 130, 220))
        self.btn_reset    = Button((px,           692, pw,      36), "Reset",
                                    color_active=(100, 20, 20))

        self.all_buttons = [
            self.btn_astar, self.btn_gbfs,
            self.btn_manh, self.btn_eucl,
            self.btn_wall, self.btn_sedit, self.btn_gedit,
            self.btn_generate, self.btn_clear,
            self.btn_run, self.btn_dynamic, self.btn_animate, self.btn_reset,
        ]
        self.all_sliders = [
            self.sl_rows, self.sl_cols, self.sl_density,
            self.sl_speed, self.sl_dynprob,
        ]

    # ── Grid helpers ───────────────────────────────────────────────────────
    def _cell_size(self):
        cw = GRID_W // self.cols
        ch = (SCREEN_H - 20) // self.rows
        return min(cw, ch, 48)

    def _grid_offset(self):
        cs = self._cell_size()
        gw = cs * self.cols
        gh = cs * self.rows
        ox = (GRID_W - gw) // 2
        oy = (SCREEN_H - gh) // 2
        return ox, oy

    def _pixel_to_cell(self, px, py):
        cs = self._cell_size()
        ox, oy = self._grid_offset()
        c = (px - ox) // cs
        r = (py - oy) // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _cell_rect(self, r, c):
        cs = self._cell_size()
        ox, oy = self._grid_offset()
        return pygame.Rect(ox + c*cs, oy + r*cs, cs, cs)

    def _resize_grid(self):
        self.rows = self.sl_rows.display_value
        self.cols = self.sl_cols.display_value
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid[0][0] = START
        self.grid[self.rows-1][self.cols-1] = GOAL
        self._clear_viz()

    def _generate_map(self):
        self._resize_grid()
        density = self.sl_density.display_value / 100.0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.start or (r, c) == self.goal:
                    continue
                if random.random() < density:
                    self.grid[r][c] = WALL
        self.status_msg   = f"Map generated ({int(density*100)}% walls)"
        self.status_color = C_SUBTEXT

    def _clear_viz(self):
        self.visited_cells  = []
        self.frontier_cells = set()
        self.path_cells     = []
        self.agent_pos      = None
        self.agent_path     = []
        self.agent_step     = 0
        self.animating      = False
        self.visit_index    = 0
        self.show_path      = False
        self.agent_running  = False
        self.nodes_visited  = 0
        self.path_cost      = 0
        self.exec_time_ms   = 0
        self.replan_count   = 0
        # Strip viz cells from grid
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (VISITED, FRONTIER, PATH, AGENT):
                    self.grid[r][c] = EMPTY
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

    def _run_search(self, from_pos=None, silent=False):
        """Run the selected algorithm. Returns (path, visited, success)."""
        h_fn = manhattan if self.heuristic == "Manhattan" else euclidean
        start = from_pos if from_pos else self.start

        # Temporarily treat VISITED/PATH cells as EMPTY for replanning
        snapshot = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (VISITED, FRONTIER, PATH, AGENT):
                    snapshot[(r,c)] = self.grid[r][c]
                    self.grid[r][c] = EMPTY
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

        t0 = time.time()
        if self.algorithm == "A*":
            path, visited, ok = astar(self.grid, start, self.goal, h_fn)
        else:
            path, visited, ok = gbfs(self.grid, start, self.goal, h_fn)
        elapsed = (time.time() - t0) * 1000

        # Restore snapshot
        for (r,c), v in snapshot.items():
            if self.grid[r][c] not in (WALL, START, GOAL):
                self.grid[r][c] = v

        if not silent:
            self.exec_time_ms  = elapsed
            self.nodes_visited = len(visited)
            self.path_cost     = len(path) - 1 if path else 0

        return path, visited, ok

    def _start_animation(self):
        """Full search + animation from scratch."""
        self._clear_viz()
        path, visited, ok = self._run_search()
        if ok:
            self.visited_cells  = visited
            self.path_cells     = path
            self.animating      = True
            self.visit_index    = 0
            self.show_path      = False
            self.status_msg     = f"{self.algorithm} running…"
            self.status_color   = C_WARNING
        else:
            self.status_msg   = "No path found!"
            self.status_color = C_ERROR

    def _start_agent(self):
        """Animate the agent walking the found path."""
        if not self.path_cells:
            self.status_msg   = "Run search first!"
            self.status_color = C_ERROR
            return
        self.agent_path    = list(self.path_cells)
        self.agent_step    = 0
        self.agent_pos     = self.agent_path[0]
        self.agent_running = True
        self.agent_timer   = 0
        self.replan_count  = 0
        self.status_msg    = "Agent moving…"
        self.status_color  = C_SUCCESS

    # ── Dynamic obstacle logic ─────────────────────────────────────────────
    def _dynamic_tick(self, dt):
        if not self.agent_running or not self.dynamic_mode:
            return
        spawn_prob = self.sl_dynprob.display_value / 100.0
        path_set = set(self.agent_path)
        spawned_on_path = False
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == EMPTY:
                    if random.random() < spawn_prob * (dt/1000):
                        self.grid[r][c] = WALL
                        if (r, c) in path_set and (r,c) != self.agent_pos:
                            spawned_on_path = True
        if spawned_on_path:
            self._replan()

    def _replan(self):
        """Replan from agent's current position."""
        current = self.agent_pos
        path, _, ok = self._run_search(from_pos=current, silent=True)
        if ok:
            self.agent_path  = path
            self.agent_step  = 0
            self.path_cells  = path
            self.replan_count += 1
            self.status_msg   = f"Replanned! (×{self.replan_count})"
            self.status_color = C_WARNING
            self._repaint_path()
        else:
            self.agent_running = False
            self.status_msg    = "Path blocked – no route!"
            self.status_color  = C_ERROR

    def _repaint_path(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == PATH:
                    self.grid[r][c] = VISITED
        for cell in self.path_cells:
            if self.grid[cell[0]][cell[1]] not in (START, GOAL, WALL, AGENT):
                self.grid[cell[0]][cell[1]] = PATH
        self.grid[self.goal[0]][self.goal[1]] = GOAL

    # ── Drawing ────────────────────────────────────────────────────────────
    def _draw_grid(self):
        cs = self._cell_size()
        ox, oy = self._grid_offset()

        cell_colors = {
            EMPTY:    C_EMPTY,
            WALL:     C_WALL,
            START:    C_START,
            GOAL:     C_GOAL,
            VISITED:  C_VISITED,
            FRONTIER: C_FRONTIER,
            PATH:     C_PATH,
            AGENT:    C_AGENT,
        }

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(ox + c*cs, oy + r*cs, cs-1, cs-1)
                ct = self.grid[r][c]
                color = cell_colors.get(ct, C_EMPTY)
                pygame.draw.rect(self.screen, color, rect, border_radius=2 if cs > 10 else 0)

                # Draw labels on START/GOAL
                if cs >= 18:
                    if ct == START:
                        draw_text(self.screen, "S", FONT_SMALL, BG,
                                  rect.centerx, rect.centery, center=True)
                    elif ct == GOAL:
                        draw_text(self.screen, "G", FONT_SMALL, BG,
                                  rect.centerx, rect.centery, center=True)
                    elif ct == AGENT:
                        draw_text(self.screen, "●", FONT_SMALL, BG,
                                  rect.centerx, rect.centery, center=True)

    def _draw_panel(self):
        px = GRID_W
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(self.screen, ACCENT, (px, 0), (px, SCREEN_H), 1)

        x = px + 14
        # Title
        draw_text(self.screen, "PATHFINDING AGENT", FONT_TITLE, ACCENT, x, 14)

        # Section: Algorithm
        draw_text(self.screen, "ALGORITHM", FONT_LABEL, C_SUBTEXT, x, 46)
        self.btn_astar.draw(self.screen)
        self.btn_gbfs.draw(self.screen)

        # Section: Heuristic
        draw_text(self.screen, "HEURISTIC", FONT_LABEL, C_SUBTEXT, x, 114)
        self.btn_manh.draw(self.screen)
        self.btn_eucl.draw(self.screen)

        # Section: Edit mode
        draw_text(self.screen, "EDIT MODE", FONT_LABEL, C_SUBTEXT, x, 182)
        self.btn_wall.draw(self.screen)
        self.btn_sedit.draw(self.screen)
        self.btn_gedit.draw(self.screen)

        # Sliders
        for sl in self.all_sliders:
            sl.draw(self.screen)

        # Action buttons
        self.btn_generate.draw(self.screen)
        self.btn_clear.draw(self.screen)
        self.btn_run.draw(self.screen)
        self.btn_dynamic.draw(self.screen)
        self.btn_animate.draw(self.screen)
        self.btn_reset.draw(self.screen)

        # Metrics dashboard
        my = 738
        pygame.draw.rect(self.screen, C_BUTTON,
                          (px+10, my, PANEL_W-20, 72), border_radius=8)
        pygame.draw.rect(self.screen, C_GRID,
                          (px+10, my, PANEL_W-20, 72), 1, border_radius=8)

        draw_text(self.screen, "METRICS", FONT_LABEL, C_SUBTEXT, px+20, my+6)
        mw = (PANEL_W-20)//3
        metrics = [
            ("Nodes", str(self.nodes_visited)),
            ("Cost",  str(self.path_cost)),
            ("Time",  f"{self.exec_time_ms:.1f}ms"),
        ]
        for i, (lbl, val) in enumerate(metrics):
            mx = px+10 + i*mw + mw//2
            draw_text(self.screen, val,  FONT_METRIC, ACCENT,   mx, my+32, center=True)
            draw_text(self.screen, lbl,  FONT_TINY,   C_SUBTEXT, mx, my+56, center=True)

        # Status message
        draw_text(self.screen, self.status_msg, FONT_LABEL,
                  self.status_color, px+14, SCREEN_H-26)

        # Dynamic replan counter
        if self.replan_count > 0:
            draw_text(self.screen, f"Re-plans: {self.replan_count}",
                      FONT_SMALL, C_WARNING, px+14, SCREEN_H-44)

        # Heuristic formula reminder
        if self.heuristic == "Manhattan":
            formula = "|x1-x2| + |y1-y2|"
        else:
            formula = "√((x1-x2)²+(y1-y2)²)"
        draw_text(self.screen, formula, FONT_TINY, C_SUBTEXT, px+14, SCREEN_H-58)

    # ── Animation step ─────────────────────────────────────────────────────
    def _step_animation(self):
        if not self.animating:
            return
        speed = max(1, self.sl_speed.display_value)
        for _ in range(speed):
            if self.visit_index < len(self.visited_cells):
                cell = self.visited_cells[self.visit_index]
                if self.grid[cell[0]][cell[1]] not in (START, GOAL):
                    self.grid[cell[0]][cell[1]] = VISITED
                self.visit_index += 1
            else:
                if not self.show_path:
                    self.show_path = True
                    for cell in self.path_cells:
                        if self.grid[cell[0]][cell[1]] not in (START, GOAL):
                            self.grid[cell[0]][cell[1]] = PATH
                    cost = len(self.path_cells)-1
                    self.status_msg   = (f"{self.algorithm} done | "
                                         f"Visited: {self.nodes_visited} | "
                                         f"Cost: {cost} | "
                                         f"{self.exec_time_ms:.1f}ms")
                    self.status_color = C_SUCCESS
                self.animating = False
                break

    def _step_agent(self, dt):
        if not self.agent_running:
            return
        self.agent_timer += dt
        if self.agent_timer < self.agent_interval:
            return
        self.agent_timer = 0

        # Clear previous agent cell
        if self.agent_pos and self.grid[self.agent_pos[0]][self.agent_pos[1]] == AGENT:
            self.grid[self.agent_pos[0]][self.agent_pos[1]] = PATH

        self.agent_step += 1
        if self.agent_step >= len(self.agent_path):
            self.agent_running = False
            self.agent_pos = self.goal
            self.grid[self.goal[0]][self.goal[1]] = GOAL
            self.status_msg   = "Agent reached the goal! 🎉"
            self.status_color = C_SUCCESS
            return

        self.agent_pos = self.agent_path[self.agent_step]
        r, c = self.agent_pos
        if self.grid[r][c] not in (GOAL, START):
            self.grid[r][c] = AGENT

    # ── Event handling ─────────────────────────────────────────────────────
    def _handle_events(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Sliders
            for sl in self.all_sliders:
                sl.handle(event)

            # Mouse on grid
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < GRID_W:
                    self.mouse_down = True
                    self._toggle_cell(event.pos)
            if event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
            if event.type == pygame.MOUSEMOTION and self.mouse_down:
                if event.pos[0] < GRID_W:
                    self._toggle_cell(event.pos)

            # Buttons
            if self.btn_astar.handle(event):
                self.algorithm = "A*"
                self.btn_astar.active = True
                self.btn_gbfs.active  = False

            if self.btn_gbfs.handle(event):
                self.algorithm = "GBFS"
                self.btn_gbfs.active  = True
                self.btn_astar.active = False

            if self.btn_manh.handle(event):
                self.heuristic = "Manhattan"
                self.btn_manh.active = True
                self.btn_eucl.active = False

            if self.btn_eucl.handle(event):
                self.heuristic = "Euclidean"
                self.btn_eucl.active = True
                self.btn_manh.active = False

            if self.btn_wall.handle(event):
                self.edit_mode = "Wall"
                self.btn_wall.active  = True
                self.btn_sedit.active = False
                self.btn_gedit.active = False

            if self.btn_sedit.handle(event):
                self.edit_mode = "Start"
                self.btn_sedit.active = True
                self.btn_wall.active  = False
                self.btn_gedit.active = False

            if self.btn_gedit.handle(event):
                self.edit_mode = "Goal"
                self.btn_gedit.active = True
                self.btn_wall.active  = False
                self.btn_sedit.active = False

            if self.btn_generate.handle(event):
                self._generate_map()

            if self.btn_clear.handle(event):
                self._resize_grid()
                self.status_msg   = "Grid cleared"
                self.status_color = C_SUBTEXT

            if self.btn_run.handle(event):
                self._start_animation()

            if self.btn_dynamic.handle(event):
                self.dynamic_mode = self.btn_dynamic.active
                lbl = "⚡ Dynamic Mode ON" if self.dynamic_mode else "⚡ Dynamic Mode OFF"
                self.btn_dynamic.label = lbl

            if self.btn_animate.handle(event):
                self._start_agent()

            if self.btn_reset.handle(event):
                self._clear_viz()
                self.status_msg   = "Visualization cleared"
                self.status_color = C_SUBTEXT

            # Keyboard shortcuts
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._start_animation()
                if event.key == pygame.K_g:
                    self._generate_map()
                if event.key == pygame.K_c:
                    self._resize_grid()
                if event.key == pygame.K_SPACE:
                    self._start_agent()

        return True

    def _toggle_cell(self, pos):
        cell = self._pixel_to_cell(*pos)
        if cell is None:
            return
        r, c = cell
        if self.edit_mode == "Wall":
            if (r, c) not in (self.start, self.goal):
                current = self.grid[r][c]
                self.grid[r][c] = WALL if current != WALL else EMPTY
        elif self.edit_mode == "Start":
            # Move start
            sr, sc = self.start
            if self.grid[sr][sc] == START:
                self.grid[sr][sc] = EMPTY
            self.start = (r, c)
            self.grid[r][c] = START
            self._clear_viz()
        elif self.edit_mode == "Goal":
            gr, gc = self.goal
            if self.grid[gr][gc] == GOAL:
                self.grid[gr][gc] = EMPTY
            self.goal = (r, c)
            self.grid[r][c] = GOAL
            self._clear_viz()

    # ── Legend ─────────────────────────────────────────────────────────────
    def _draw_legend(self):
        items = [
            (C_START,    "Start (S)"),
            (C_GOAL,     "Goal (G)"),
            (C_WALL,     "Wall"),
            (C_VISITED,  "Visited"),
            (C_FRONTIER, "Frontier"),
            (C_PATH,     "Path"),
            (C_AGENT,    "Agent"),
        ]
        lx, ly = 10, SCREEN_H - 28
        for color, label in items:
            pygame.draw.rect(self.screen, color, (lx, ly, 12, 12), border_radius=2)
            draw_text(self.screen, label, FONT_TINY, C_SUBTEXT, lx+16, ly)
            lx += 80

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)

            running = self._handle_events(dt)

            # Update animation
            self._step_animation()
            self._step_agent(dt)
            self._dynamic_tick(dt)

            # Sync slider values (for live updates before generating)
            # (Resize only happens on button press, not live, to avoid flicker)

            # Draw
            self.screen.fill(BG)
            self._draw_grid()
            self._draw_legend()
            self._draw_panel()

            pygame.display.flip()

        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PathfindingApp()
    app.run()
