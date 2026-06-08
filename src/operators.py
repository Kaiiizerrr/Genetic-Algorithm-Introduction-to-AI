import random

from src.structures import Location, Route


# ══════════════════════════════════════════════════════════════════════════════
# Cycle Crossover  (CX)
# ══════════════════════════════════════════════════════════════════════════════

def cycle_crossover(parent1: Route, parent2: Route) -> tuple:
    """
    Cycle Crossover (CX) operator.

    Produces two valid offspring by detecting positional cycles between the
    two parent routes and alternately assigning each cycle to opposite parents.
    Because every gene belongs to exactly one cycle, both offspring are
    guaranteed to contain every location exactly once.

    Algorithm
    ---------
    1.  Initialise ``child1_genes`` and ``child2_genes`` — two gene lists of
        size 10 filled with ``None``.
    2.  Build a position look-up for parent2:
        ``p2_pos[location_id] = index_in_parent2``.
    3.  Repeat until every index is marked visited:
        a.  Start at the first unvisited index.
        b.  Trace the cycle:
            - Record the current index; mark it visited.
            - Follow the link: look up where parent1's gene at the current
              position appears inside parent2 → that index becomes ``current``.
            - Repeat until the path loops back to the start position.
        c.  Assign the cycle's genes to the children, alternating each cycle:
            - **Even** cycles  → child1 ← parent1 genes; child2 ← parent2 genes.
            - **Odd**  cycles  → child1 ← parent2 genes; child2 ← parent1 genes.
    4.  Construct Route objects and run integrity assertions.

    Edge-Case Handling
    ------------------
    *   **Cycle of length 1** — occurs when ``parent1[i] == parent2[i]``
        (same location at the same index in both parents).  The single-index
        cycle is detected naturally: the jump immediately lands back on the
        start, so the while-loop exits after one iteration.  The gene is
        assigned according to the current cycle parity — no special branching
        needed.
    *   **Identical parents** — every index forms its own length-1 cycle; all
        even-parity cycles are assigned from parent1, so child1 == parent1
        and child2 == parent2.  Both results are valid permutations.

    Parameters
    ----------
    parent1 : Route  First parent solution.
    parent2 : Route  Second parent solution.

    Returns
    -------
    tuple[Route, Route]
        ``(child1, child2)`` — two offspring routes.
    """

    SIZE: int = 10

    # ── Step 1 : Initialise two child gene arrays ─────────────────────────────
    child1_genes: list = [None] * SIZE
    child2_genes: list = [None] * SIZE

    p1: list = parent1.vertices   # shorthand references
    p2: list = parent2.vertices

    # ── Step 2 : Position look-up for parent2 ─────────────────────────────────
    # Maps each location's id → its array index inside parent2.
    # Used to "jump" from a gene value back to its position in parent2.
    p2_pos: dict = {p2[i].id: i for i in range(SIZE)}

    # ── Step 3 : Cycle detection and alternating gene assignment ───────────────
    visited:   list = [False] * SIZE
    cycle_num: int  = 0            # tracks even / odd parity of the current cycle

    while False in visited:

        # ── 3a : Find the first unvisited index to seed this cycle ────────────
        start:         int  = visited.index(False)
        cycle_indices: list = []
        current:       int  = start

        # ── 3b : Trace the cycle until we return to start ─────────────────────
        #
        #   Loop body:
        #     1. Record the current position in this cycle.
        #     2. Mark it as visited so we don't re-enter it.
        #     3. Jump: find the index inside parent2 where parent1's gene
        #        at `current` is located → that becomes the next `current`.
        #   The loop exits when `current` arrives back at an already-visited
        #   index (== start), signalling the cycle has closed.
        #
        while not visited[current]:
            cycle_indices.append(current)
            visited[current] = True
            # Jump to the index in parent2 that holds parent1's gene at `current`
            current = p2_pos[p1[current].id]

        # ── 3c : Assign genes; alternate which parent "owns" each cycle ────────
        if cycle_num % 2 == 0:
            # Even cycle → child1 inherits from parent1; child2 from parent2
            for idx in cycle_indices:
                child1_genes[idx] = p1[idx]
                child2_genes[idx] = p2[idx]
        else:
            # Odd cycle → child1 inherits from parent2; child2 from parent1
            for idx in cycle_indices:
                child1_genes[idx] = p2[idx]
                child2_genes[idx] = p1[idx]

        cycle_num += 1

    # ── Step 4 : Build Route objects ──────────────────────────────────────────
    child1 = Route(child1_genes)
    child2 = Route(child2_genes)

    # ── Integrity validation ──────────────────────────────────────────────────
    # Guarantees that neither child contains duplicate or missing locations.
    # Because Location.__hash__ / __eq__ are keyed on .id, the set operation
    # correctly detects any repeated location object.
    assert len(set(child1.vertices)) == SIZE, (
        f"CX integrity error - child1 has duplicate/missing locations: "
        f"{[v.id for v in child1.vertices]}"
    )
    assert len(set(child2.vertices)) == SIZE, (
        f"CX integrity error - child2 has duplicate/missing locations: "
        f"{[v.id for v in child2.vertices]}"
    )

    return child1, child2


# ══════════════════════════════════════════════════════════════════════════════
# Swap Mutation
# ══════════════════════════════════════════════════════════════════════════════

def swap_mutation(route: Route, mutation_rate: float) -> Route:
    """
    Swap Mutation operator.

    With probability ``mutation_rate``, randomly selects two distinct indices
    within the route's vertex array and swaps the Location objects at those
    positions in-place.

    A swap preserves the route's permutation validity — only the order of
    two stops changes, so no location is added or removed.

    Parameters
    ----------
    route         : Route  The route to (possibly) mutate.
    mutation_rate : float  Trigger probability in the range ``[0.0, 1.0]``.

    Returns
    -------
    Route
        The same Route object, either mutated or unchanged.
    """

    # ── Probability gate ──────────────────────────────────────────────────────
    if random.random() < mutation_rate:

        # Draw two distinct random indices from [0, 9]
        idx1, idx2 = random.sample(range(10), 2)

        # Swap the Location objects at the two positions in-place
        route.vertices[idx1], route.vertices[idx2] = (
            route.vertices[idx2],
            route.vertices[idx1],
        )

    return route
