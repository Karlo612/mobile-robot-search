import time
import tracemalloc
import numpy as np
import copy
from dataclasses import dataclass
from typing import List, Dict

from CodeBase.Benchmark.grid_factory import generate_grid
from CodeBase.Search.Astar_graph_based import AStarPlanner_graphbased
from CodeBase.Search.Astar_tree_based import AStarPlanner_treebased
from CodeBase.Search.bfs import BFSPlanner_graphbased, BFSPlanner_treesearch
from CodeBase.Search.dfs import DFSPlanner_graphbased, DFSPlanner_treebased
from CodeBase.Benchmark.plotter import plot_all_tests


# ---------------------------------------------------------
# Planner registry 
# ---------------------------------------------------------
PLANNERS = {
    "A* Graph": AStarPlanner_graphbased,
    "A* Tree":  AStarPlanner_treebased,
    "BFS Graph": BFSPlanner_graphbased,
    "BFS Tree":  BFSPlanner_treesearch,
    "DFS Graph": DFSPlanner_graphbased,
    "DFS Tree":  DFSPlanner_treebased,
}

@dataclass
class ResultRow:
    """Flat row just for text table / quick summary."""
    test_title: str
    planner_name: str
    runtime: float
    expansions: int
    path_length: int
    memory_kb: float


class Comparator:
    def __init__(self, robot_radius=1.0, motion_model="8n", num_runs=5):
        self.results: List[ResultRow] = []   # flat table-style results
        self.test_runs: List[Dict] = []      # structured per-test data for plotting
        self.robot_radius = robot_radius
        self.motion_model = motion_model
        self.num_runs = num_runs  # Number of runs to average

    # ---------------------------------------------------------
    # Main benchmark runner
    # ---------------------------------------------------------
    def run_all_tests(self):

        tests = [
            ("Small 10x10", 10),
            # ("Medium 50x50", 50),  # COMMENT THIS FOR NOW
            # ("Large 100x100", 100),
        ]

        for title, size in tests:
            print(f"\n=== Running benchmark on {title} ===")

            # Connectivity-aware grid generation: regenerate until BFS finds a path
            max_attempts = 20
            grid_map = None
            start = None
            goal = None
            obstacles = None
            
            for attempt in range(max_attempts):
                # Generate grid + inflated obstacles + random start/goal
                grid_map, start, goal, obstacles = generate_grid(size, self.robot_radius)
                
                # Quick connectivity check: run BFS Graph once
                grid_copy = copy.deepcopy(grid_map)
                bfs_checker = BFSPlanner_graphbased(grid_copy, motion_model=self.motion_model, visualizer=None)
                test_path = bfs_checker.plan(start, goal)
                
                if test_path and test_path[-1] == goal:
                    print(f"  Generated valid grid (attempt {attempt + 1})")
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(f"  Regenerating grid (attempt {attempt + 1}/{max_attempts}) - no path found")
                    else:
                        print(f"  Warning: Could not generate valid grid after {max_attempts} attempts")
                        # Use the last generated grid anyway (will show as FAILED)

            # Per-test structure for plotting
            per_test_data = {
                "title": title,
                "size": size,
                "start": start,
                "goal": goal,
                "radius": self.robot_radius,
                "obstacles": [(o.gx, o.gy) for o in obstacles],
                "resolution": grid_map.resolution,
                "grid_height": grid_map.height,
                "grid_width": grid_map.width,
                "planners": {}   # filled below
            }

            for planner_name, PlannerClass in PLANNERS.items():
                # Use fewer runs for larger grids to save time
                actual_runs = 1 if size >= 50 else self.num_runs
                print(f"  -> {planner_name} (averaging over {actual_runs} runs)")

                # Set max_expansions for DFS Tree (always explicit so we can label CAPPED vs FAILED)
                max_expansions = None
                if planner_name == "DFS Tree":
                    max_expansions = 20_000 if size < 50 else 50_000
                    print(f"      (max_expansions cap: {max_expansions:,})")

                # Run multiple times and collect metrics
                runtimes = []
                memories = []
                expansions_list = []
                path_lens = []
                heatmap = None  # Store only first run's heatmap
                paths = []
                reached_goals = []
                capped_flags = []  # Track if run hit max_expansions cap

                for run_idx in range(actual_runs):
                    grid_copy = copy.deepcopy(grid_map)
                    # Instantiate planner (no visualizer in benchmark mode)
                    if planner_name == "DFS Tree":
                        planner = PlannerClass(
                            grid_copy,
                            motion_model=self.motion_model,
                            visualizer=None,
                            max_expansions=max_expansions,
                        )
                    else:
                        planner = PlannerClass(grid_copy, motion_model=self.motion_model, visualizer=None)

                    # Measure time + memory
                    tracemalloc.start()
                    t0 = time.time()
                    path = planner.plan(start, goal)
                    t1 = time.time()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    runtime = t1 - t0
                    memory_kb = peak / 1024.0
                    expansions = getattr(planner, "expanded_count", -1)
                    path_len = len(path) if path else 0
                    reached_goal = bool(path and path[-1] == goal)
                    
                    # Debug prints disabled for final benchmark run
                    # if run_idx == 0:  # Only print for first run
                    #     print(f"[DEBUG] {planner_name}: reached_goal={reached_goal}, path_len={path_len}, path_end={path[-1] if path else None}")
                    
                    # Check if run was capped (DFS Tree explicitly flags when it hits the cap)
                    was_capped = getattr(planner, "was_capped", False)

                    # Build heatmap from planner.expansion_map only for first run
                    if run_idx == 0:
                        h, w = grid_map.height, grid_map.width
                        heat = np.zeros((h, w), dtype=int)
                        expansion_map = getattr(planner, "expansion_map", {})

                        for (x, y), count in expansion_map.items():
                            if 0 <= x < w and 0 <= y < h:
                                heat[y, x] = count
                        heatmap = heat

                    runtimes.append(runtime)
                    memories.append(memory_kb)
                    expansions_list.append(expansions)
                    # Only include path_len if goal was reached (don't average zeros from failures)
                    path_lens.append(path_len if reached_goal else None)
                    paths.append(path if path else [])
                    reached_goals.append(reached_goal)
                    capped_flags.append(was_capped)

                # Determine status: all_capped, all_failed, or success
                all_capped = all(capped_flags)
                all_failed = not any(reached_goals) and not all_capped
                any_success = any(reached_goals)
                
                # Compute averages - only over successful runs for path_len
                avg_runtime = np.mean(runtimes)
                avg_memory = np.mean(memories)
                
                # For expansions, filter out -1 (invalid) values before averaging
                valid_expansions = [e for e in expansions_list if e >= 0]
                avg_expansions = np.mean(valid_expansions) if valid_expansions else -1
                
                # Path length: average only successful runs, or mark as failed/capped
                successful_path_lens = [pl for pl in path_lens if pl is not None and pl > 0]
                if successful_path_lens:
                    avg_path_len = np.mean(successful_path_lens)
                elif all_capped:
                    avg_path_len = -2  # Special marker for "capped"
                else:
                    avg_path_len = -1  # Special marker for "failed"
                
                # Use heatmap from first run (or empty if not available)
                if heatmap is None:
                    h, w = grid_map.height, grid_map.width
                    avg_heatmap = np.zeros((h, w), dtype=int)
                else:
                    avg_heatmap = heatmap
                
                # Use the path from the first successful run, or empty if none succeeded
                best_path = None
                for p in paths:
                    if p:
                        best_path = p
                        break

                # Store status information
                status = ""
                if all_capped:
                    status = " (CAPPED)"
                elif all_failed:
                    status = " (FAILED)"
                elif not any_success:
                    status = " (NO PATH)"

                # Store flat row for text summary (averaged values)
                self.results.append(
                    ResultRow(
                        test_title=title,
                        planner_name=planner_name + status,
                        runtime=avg_runtime,
                        expansions=int(avg_expansions) if avg_expansions >= 0 else -1,
                        path_length=int(round(avg_path_len)) if avg_path_len >= 0 else avg_path_len,
                        memory_kb=avg_memory,
                    )
                )

                # Store structured stats for plotting (averaged values)
                per_test_data["planners"][planner_name] = {
                    "runtime": avg_runtime,
                    "expansions": int(avg_expansions) if avg_expansions >= 0 else -1,
                    "path_len": int(round(avg_path_len)) if avg_path_len >= 0 else avg_path_len,
                    "memory_kb": avg_memory,
                    "heatmap": avg_heatmap,
                    "reached_goal": any_success,
                    "was_capped": all_capped,  # all runs capped
                    "status": status.strip(),
                    "path": best_path if best_path else []
                }

            # Add this test to list of runs
            self.test_runs.append(per_test_data)

        return self.test_runs

    # ---------------------------------------------------------
    # Console print
    # ---------------------------------------------------------
    def print_results(self):
        print(f"\n====== Benchmark Results ======")
        print(f"{'Test':<20} {'Planner':<25} {'Runtime (s)':<15} {'Expansions':<12} {'Path Len':<12} {'Memory (KB)':<12}")
        print("-" * 110)
        for r in self.results:
            exp_str = str(r.expansions) if r.expansions >= 0 else "N/A"
            if r.path_length == -2:
                path_str = "CAPPED"
            elif r.path_length == -1:
                path_str = "FAILED"
            else:
                path_str = str(r.path_length)
            print(
                f"{r.test_title:<20} {r.planner_name:<25} {r.runtime:<15.4f} {exp_str:<12} {path_str:<12} {r.memory_kb:<12.1f}"
            )

    # ---------------------------------------------------------
    # Orchestrate: run + print + plot
    # ---------------------------------------------------------
    def run_and_plot(self):
        test_runs = self.run_all_tests()
        self.print_results()
        # plot_all_tests(test_runs)  # Temporarily disabled to avoid memory blow-up during benchmarking