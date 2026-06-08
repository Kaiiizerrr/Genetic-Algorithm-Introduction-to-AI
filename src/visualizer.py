from pathlib import Path

import matplotlib.pyplot as plt


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_generation_vs_best(generation_data, filename="convergence.png"):
    """Plot generation vs best distance and save the figure."""
    generations = [record["generation"] for record in generation_data]
    best_distances = [record["best"] for record in generation_data]

    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(generations, best_distances, marker="o", linestyle="-", color="#2a9d8f")
    plt.title("Generation vs Best Distance")
    plt.xlabel("Generation")
    plt.ylabel("Best Distance")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    output_path = ASSETS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_convergence_graph(history, filename="convergence_graph.png"):
    """Save the convergence graph for generation cost history."""
    generations = [record["generation"] for record in history]
    best = [record["best"] for record in history]
    avg = [record["avg"] for record in history]
    worst = [record["worst"] for record in history]

    plt.figure(figsize=(12, 7), dpi=200)
    plt.plot(generations, best, marker="o", linestyle="-", label="Best", color="#2a9d8f")
    plt.plot(generations, avg, marker="s", linestyle="--", label="Average", color="#e9c46a")
    plt.plot(generations, worst, marker="^", linestyle="-.", label="Worst", color="#e76f51")

    plt.title("GA Convergence: Generation vs Path Cost")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = ASSETS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_locations_and_best_route(locations, best_route, filename="best_route.png"):
    """Plot city coordinates and the best route path, then save the figure."""
    xs = [location.x for location in locations]
    ys = [location.y for location in locations]

    route_xs = [location.x for location in best_route.vertices] + [best_route.vertices[0].x]
    route_ys = [location.y for location in best_route.vertices] + [best_route.vertices[0].y]
    route_labels = [
        f"{location.id}: {location.name}" if getattr(location, "name", None) else str(location.id)
        for location in best_route.vertices
    ]

    plt.figure(figsize=(10, 10), dpi=200)
    plt.scatter(xs, ys, color="#264653", s=100, edgecolor="#000000", zorder=3)
    plt.plot(route_xs, route_ys, linestyle="-", color="#e76f51", linewidth=2, marker="o", markersize=6)

    for label, x, y in zip(route_labels, route_xs[:-1], route_ys[:-1]):
        plt.text(x, y, label, fontsize=8, fontweight="bold", ha="right", va="bottom")

    plt.title("Best Route Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = ASSETS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def save_route_map(best_route, filename="route_map.png"):
    """Save the spatial route map for the best route found."""
    xs = [location.x for location in best_route.vertices] + [best_route.vertices[0].x]
    ys = [location.y for location in best_route.vertices] + [best_route.vertices[0].y]
    labels = [
        f"{location.id}: {location.name}" if getattr(location, "name", None) else str(location.id)
        for location in best_route.vertices
    ]

    plt.figure(figsize=(10, 10), dpi=200)
    plt.plot(xs, ys, marker="o", linestyle="-", color="#264653", linewidth=2)
    plt.scatter(xs[:-1], ys[:-1], color="#f4a261", s=120, edgecolor="#000000", zorder=3)

    for label, x, y in zip(labels, xs[:-1], ys[:-1]):
        plt.text(x, y, label, fontsize=8, fontweight="bold", ha="right", va="bottom")

    plt.title("Best Route Spatial Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    output_path = ASSETS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def visualize(history, best_route, output_dir=None):
    """Generate and save both convergence and route map charts."""
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = ASSETS_DIR

    convergence_file = save_convergence_graph(history, filename=output_path / "convergence_graph.png")
    route_file = save_route_map(best_route, filename=output_path / "route_map.png")
    return convergence_file, route_file
