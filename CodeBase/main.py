# main.py
from .navigation_system import NavigationSystem
import json

def main():
    with open("Data/configs/config.json") as f:
        cfg = json.load(f)

    mode = cfg.get("mode", "nav").lower()

    if mode == "nav":
        system = NavigationSystem(cfg)
        system.run()

    elif mode == "bench":
        # Import Comparator only in bench mode to avoid importing plotter.py
        # which would set matplotlib backend to non-interactive
        from .Benchmark.comparator import Comparator
        print("\n=== Running Benchmark Mode ===")
        robot_r = cfg.get("robot_radius", 1.0)
        motion  = cfg.get("motion", "8n")
        cmp = Comparator(robot_radius=robot_r, motion_model=motion)
        cmp.run_and_plot()

    else:
        raise ValueError(f"Unknown config mode: {mode}")

if __name__ == "__main__":
    main()