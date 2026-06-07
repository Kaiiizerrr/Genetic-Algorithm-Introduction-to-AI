"""
structures.py  —  Member 2 Deliverable
=======================================
Data-model definitions for the TSP Genetic Algorithm.

Classes
-------
Location  — Represents a single Jollibee branch (id, x, y).
Route     — Represents a candidate solution (chromosome); an ordered
            sequence of 10 Location stops with distance / fitness tracking.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Location
# ══════════════════════════════════════════════════════════════════════════════

class Location:
    """
    Represents a single geographic stop (Jollibee branch) in the TSP.

    Attributes
    ----------
    id : int    Unique integer identifier that matches the dataset (1–10).
    x  : float  X-coordinate on the spatial grid.
    y  : float  Y-coordinate on the spatial grid.
    """

    def __init__(self, id: int, x: float, y: float) -> None:
        self.id: int   = id
        self.x:  float = float(x)
        self.y:  float = float(y)

    # ── Equality and hashing based on ID ─────────────────────────────────────
    # Required so Location objects can be stored in sets and used as dict keys.
    # The CX integrity assertion `len(set(child.vertices)) == 10` depends on this.

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Location):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Location(id={self.id}, x={self.x}, y={self.y})"


# ══════════════════════════════════════════════════════════════════════════════
# Route
# ══════════════════════════════════════════════════════════════════════════════

class Route:
    """
    Represents a candidate solution (chromosome) for the TSP.

    A Route is an ordered sequence of exactly 10 Location objects that
    defines a closed tour (the tour returns to the starting location after
    the last stop).

    Attributes
    ----------
    vertices : list[Location]
        Ordered array of 10 Location stops — the chromosome sequence.
    length   : int
        Number of stops in the route; always 10 for this problem.
    distance : float
        Total Euclidean distance of the closed tour.
        Populated by :meth:`calculate_total_distance`.
    fitness  : float
        Fitness score; defined as ``1 / distance`` so shorter tours score
        higher.  Populated by :meth:`calculate_total_distance`.
    """

    def __init__(self, vertices: list) -> None:
        self.vertices: list  = vertices      # ordered array of 10 Location objects
        self.length:   int   = len(vertices) # integer tracking the sequence length
        self.distance: float = 0.0           # total tour distance (float)
        self.fitness:  float = 0.0           # inverse-distance fitness (float)

    # ── Distance / fitness calculation ────────────────────────────────────────

    def calculate_total_distance(self, distance_matrix: list) -> float:
        """
        Sums all edge costs along the closed tour using Member 3's pre-computed
        Euclidean distance matrix.

        The tour is closed, meaning the cost of the edge from the last stop
        back to the first stop is also included.

        Parameters
        ----------
        distance_matrix : list[list[float]]
            A 10×10 grid supplied by Member 3's ``matrix_generator.py``.
            ``distance_matrix[i][j]`` holds the Euclidean distance between
            the location with ``id = i + 1`` and the location with
            ``id = j + 1`` (0-based indexing into the matrix).

        Returns
        -------
        float
            Total route distance.  The value is also stored in
            ``self.distance`` and ``self.fitness`` is updated in-place.
        """
        total: float = 0.0
        n: int = self.length

        for i in range(n):
            # Convert 1-based location id → 0-based matrix index
            from_idx: int = self.vertices[i].id - 1
            to_idx:   int = self.vertices[(i + 1) % n].id - 1
            total += distance_matrix[from_idx][to_idx]

        self.distance = total
        # Fitness: shorter distance → higher fitness
        self.fitness = 1.0 / self.distance if self.distance > 0 else float("inf")
        return self.distance

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        path = [loc.id for loc in self.vertices]
        return (
            f"Route("
            f"path={path}, "
            f"distance={self.distance:.4f}, "
            f"fitness={self.fitness:.6f})"
        )
