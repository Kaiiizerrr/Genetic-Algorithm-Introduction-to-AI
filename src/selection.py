import random


def tournament_selection(population):
    """Select a parent using tournament selection over 3 random routes."""
    if len(population) < 3:
        raise ValueError("Population must contain at least 3 routes for tournament selection.")

    tournament = random.sample(population, 3)
    winner = min(tournament, key=lambda route: route.distance)
    return winner
