import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.matrix_generator import load_locations_and_matrix
from src.operators import cycle_crossover, swap_mutation
from src.selection import tournament_selection
from src.structures import Route
from src.visualizer import plot_generation_vs_best, plot_locations_and_best_route


def create_random_route(locations, distance_matrix):
    """Create a random Route from loaded locations and calculate its distance."""
    vertices = random.sample(locations, len(locations))
    route = Route(vertices)
    route.calculate_total_distance(distance_matrix)
    return route


def initialize_population(locations, distance_matrix, population_size=50):
    """Initialize a population of random routes using the distance matrix."""
    return [create_random_route(locations, distance_matrix) for _ in range(population_size)]


def load_problem_data(filename="CX Group 5.xlsx"):
    """Load locations and the Euclidean distance matrix from Excel."""
    locations, distance_matrix = load_locations_and_matrix(filename)
    return locations, distance_matrix


def evaluate_population(population):
    """Calculate statistics for the current population."""
    distances = [route.distance for route in population]
    best = min(distances)
    worst = max(distances)
    average = sum(distances) / len(distances)
    return best, worst, average


def route_signature(route):
    """Return a hashable signature for a route based on ordered location IDs."""
    return tuple(location.id for location in route.vertices)


def run_genetic_algorithm(
    locations,
    distance_matrix,
    population_size=50,
    generations=100,
    mutation_rate=0.05,
):
    """Run the GA for a fixed number of generations and return history."""
    population = initialize_population(locations, distance_matrix, population_size)
    history = []

    for generation in range(1, generations + 1):
        new_population = []
        seen_signatures = set()

        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = cycle_crossover(parent1, parent2)

            swap_mutation(child1, mutation_rate)
            swap_mutation(child2, mutation_rate)

            child1.calculate_total_distance(distance_matrix)
            child2.calculate_total_distance(distance_matrix)

            for child in (child1, child2):
                if len(new_population) >= population_size:
                    break

                signature = route_signature(child)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    new_population.append(child)

        population = new_population
        best, worst, average = evaluate_population(population)
        history.append(
            {
                "generation": generation,
                "best": best,
                "avg": average,
                "worst": worst,
            }
        )

    return population, history


def save_generation_history_csv(history, filename="generational_history.csv"):
    """Save generation tracking history to a CSV file."""
    fieldnames = ["generation", "best", "avg", "worst"]
    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    locations, distance_matrix = load_problem_data()
    final_population, history = run_genetic_algorithm(locations, distance_matrix)
    save_generation_history_csv(history)

    best_route = min(final_population, key=lambda route: route.distance)
    convergence_file = plot_generation_vs_best(history, filename="convergence.png")
    route_file = plot_locations_and_best_route(locations, best_route, filename="best_route.png")

    print("Generation history:")
    for record in history:
        print(record)
    print("\nBest route at end:")
    print(best_route)
    print(f"Convergence plot saved to: {convergence_file}")
    print(f"Best route plot saved to: {route_file}")
