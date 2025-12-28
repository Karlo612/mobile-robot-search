from .planner import Planner


class DFSPlanner_treebased(Planner):
    """
    Depth-First Search (tree search):
    - LIFO stack
    - NO visited/closed set (true tree search)
    - Deterministic neighbour order (reverse-push)
    - max_expansions guard to prevent infinite looping on cyclic grids
    - visit_id-based parenting so repeated states can still reconstruct a valid path
    """

    def __init__(self, grid_map, motion_model="8n", visualizer=None, max_expansions=200_000):
        super().__init__(grid_map, motion_model, visualizer)
        self.max_expansions = max_expansions
        self.expanded_count = 0
        self.expansion_map = {}
        self._vid = 0
        self.was_capped = False

    def get_neighbors(self, gx, gy):
        if self.motion_model == "4n":
            moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            moves = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

        for dx, dy in moves:
            nx, ny = gx + dx, gy + dy

            if not self.grid_map.is_inside(nx, ny):
                continue
            if self.grid_map.is_obstacle(nx, ny):
                continue
            if self.grid_map.is_inflated(nx, ny):
                if self.visualizer:
                    self.visualizer.draw_inflated(nx, ny)
                continue

            yield (nx, ny)

    def plan(self, start, goal):
        # reset per-run state (reproducible benchmarks)
        self.expanded_count = 0
        self.expansion_map.clear()
        self._vid = 0
        self.was_capped = False

        vis = self.visualizer
        if vis:
            vis.draw_start_goal(start, goal)
            vis.update()

        # stack holds (state, visit_id)
        stack = []
        parent = {}  # key: (state, vid) -> (parent_state, parent_vid) or None for root

        self._vid += 1
        root_key = (start, self._vid)
        parent[root_key] = None
        stack.append(root_key)

        while stack:
            if self.expanded_count >= self.max_expansions:
                self.was_capped = True
                return None  # safety cap hit (tree DFS can loop on cycles)

            (v, vid) = stack.pop()
            cx, cy = v

            # metrics
            self.expanded_count += 1
            self.expansion_map[(cx, cy)] = self.expansion_map.get((cx, cy), 0) + 1

            if vis:
                vis.draw_explored(cx, cy)
                vis.update()

            if v == goal:
                return self._reconstruct_path(parent, (v, vid))

            neighbors = list(self.get_neighbors(cx, cy))
            for child in reversed(neighbors):  # fixed DFS ordering
                self._vid += 1
                child_key = (child, self._vid)
                parent[child_key] = (v, vid)
                stack.append(child_key)

                if vis:
                    vis.draw_frontier(child[0], child[1])

        return None

    def _reconstruct_path(self, parent, goal_key):
        # unwind (state, vid) chain back to root
        out = []
        cur = goal_key
        while cur is not None:
            state, _vid = cur
            out.append(state)
            cur = parent.get(cur, None)
        out.reverse()
        return out