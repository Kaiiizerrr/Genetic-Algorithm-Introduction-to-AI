"""
test_member2.py
===============
Run from the project root:
    python test_member2.py

Tests every requirement in the PRD for Member 2:
  - Location class (id, x, y, equality, hashing)
  - Route class (vertices, length, distance, fitness,
                 calculate_total_distance)
  - cycle_crossover (valid offspring, no duplicates, edge cases)
  - swap_mutation (probability gate, exactly 2 positions swap)
"""

import math
import random

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from src.structures import Location, Route
    from src.operators  import cycle_crossover, swap_mutation
    print("Imports OK\n")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Make sure you run this from the project root folder.")
    raise SystemExit(1)


# ══ Helpers ══════════════════════════════════════════════════════════════════

def make_locations():
    """Return the 10 Jollibee locations from the PRD dataset."""
    data = [
        (1,  1, 3), (2,  2, 8), (3,  3, 7), (4,  4, 6), (5,  4, 2),
        (6,  6, 5), (7,  6, 1), (8,  7, 4), (9,  8, 3), (10, 9, 2),
    ]
    return [Location(id_, x, y) for id_, x, y in data]


def make_distance_matrix(locations):
    """Build a real 10x10 Euclidean distance matrix."""
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = locations[i].x - locations[j].x
            dy = locations[i].y - locations[j].y
            matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    return matrix


def section(title):
    print(f"{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def ok(msg):  print(f"Ok {msg}")
def fail(msg): print(f"Fail {msg}")


# ══ Test 1 — Location class ═══════════════════════════════════════════════════

section("TEST 1 Location class")

locations = make_locations()

# Attributes
loc = locations[0]
assert isinstance(loc.id, int),   "id must be int"
assert isinstance(loc.x,  float), "x must be float"
assert isinstance(loc.y,  float), "y must be float"
ok("id is int, x and y are float")

# Values match the PRD table
assert loc.id == 1 and loc.x == 1.0 and loc.y == 3.0
ok(f"Location 1 values correct -> {loc}")

# Equality
loc_copy = Location(1, 99, 99)   # same id, different coords
assert loc == loc_copy,   "Locations with same id must be equal"
assert loc != locations[1], "Locations with different id must not be equal"
ok("__eq__ works (comparison by id)")

# Hashability — required for set(child.vertices) integrity check
loc_set = set(locations)
assert len(loc_set) == 10, "All 10 locations should be unique in a set"
ok("__hash__ works - all 10 locations unique in a set")


# ══ Test 2 — Route class ══════════════════════════════════════════════════════

section("TEST 2 - Route class")

dist_matrix = make_distance_matrix(locations)
route = Route(locations[:])   # copy of the list

# Attributes exist and have correct types
assert isinstance(route.vertices, list), "vertices must be a list"
assert isinstance(route.length,   int),  "length must be int"
assert isinstance(route.distance, float),"distance must be float"
assert isinstance(route.fitness,  float),"fitness must be float"
ok("vertices (list), length (int), distance (float), fitness (float) all present")

# Length
assert route.length == 10, f"length should be 10, got {route.length}"
ok("length == 10")

# calculate_total_distance
returned = route.calculate_total_distance(dist_matrix)
assert returned == route.distance, "return value must equal self.distance"
assert route.distance > 0,         "distance must be > 0"
assert abs(route.fitness - 1.0 / route.distance) < 1e-9, \
    "fitness must equal 1 / distance"
ok(f"calculate_total_distance -> {route.distance:.4f}")
ok(f"fitness == 1/distance  -> {route.fitness:.6f}")

# Closed tour: going 1→2→…→10→1 must be longer than 0
assert route.distance > 0
ok("Closed tour cost is positive")


# ══ Test 3 — Cycle Crossover: basic correctness ═══════════════════════════════

section("TEST 3 cycle_crossover - basic correctness")

random.seed(42)
s1 = locations[:]
s2 = locations[:]
random.shuffle(s1)
random.shuffle(s2)

parent1 = Route(s1)
parent2 = Route(s2)

print(f"  Parent 1: {[v.id for v in parent1.vertices]}")
print(f"  Parent 2: {[v.id for v in parent2.vertices]}")

result = cycle_crossover(parent1, parent2)
assert isinstance(result, tuple) and len(result) == 2, \
    "cycle_crossover must return a tuple of 2 Route objects"
child1, child2 = result

assert isinstance(child1, Route), "child1 must be a Route"
assert isinstance(child2, Route), "child2 must be a Route"
ok("Returns a tuple of 2 Route objects")

print(f"  Child 1:  {[v.id for v in child1.vertices]}")
print(f"  Child 2:  {[v.id for v in child2.vertices]}")

# Integrity assertion from the PRD
assert len(set(child1.vertices)) == 10, "child1 has duplicate/missing locations"
assert len(set(child2.vertices)) == 10, "child2 has duplicate/missing locations"
ok("len(set(child.vertices)) == 10 for both children <- PRD assertion")

# Children contain all 10 IDs exactly once
ids1 = sorted(v.id for v in child1.vertices)
ids2 = sorted(v.id for v in child2.vertices)
assert ids1 == list(range(1, 11)), f"child1 ids wrong: {ids1}"
assert ids2 == list(range(1, 11)), f"child2 ids wrong: {ids2}"
ok("Both children contain IDs 1-10 with no duplicates or gaps")

# Children can calculate distance
child1.calculate_total_distance(dist_matrix)
child2.calculate_total_distance(dist_matrix)
ok(f"child1 distance={child1.distance:.4f}  child2 distance={child2.distance:.4f}")


# ══ Test 4 — Cycle Crossover: edge cases ══════════════════════════════════════

section("TEST 4 cycle_crossover - edge cases")

# Edge case A: identical parents (every cycle has length 1)
identical = Route(locations[:])
c1, c2 = cycle_crossover(identical, Route(locations[:]))
assert len(set(c1.vertices)) == 10
assert len(set(c2.vertices)) == 10
ok("Identical parents -> both children still valid (length-1 cycle edge case)")

# Edge case B: reversed parent
rev = Route(list(reversed(locations)))
c1, c2 = cycle_crossover(Route(locations[:]), rev)
assert len(set(c1.vertices)) == 10
assert len(set(c2.vertices)) == 10
ok("Reversed parent -> both children still valid")

# Edge case C: 100 random pairs — all must pass integrity
random.seed(0)
for trial in range(100):
    a, b = locations[:], locations[:]
    random.shuffle(a); random.shuffle(b)
    c1, c2 = cycle_crossover(Route(a), Route(b))
    assert len(set(c1.vertices)) == 10, f"Trial {trial}: child1 integrity fail"
    assert len(set(c2.vertices)) == 10, f"Trial {trial}: child2 integrity fail"
ok("100 random parent pairs -> all children pass integrity  (stress test)")


# ══ Test 5 — Swap Mutation ════════════════════════════════════════════════════

section("TEST 5 - swap_mutation")

random.seed(7)
test_route = Route(locations[:])

# Always mutate (rate = 1.0)
before = [v.id for v in test_route.vertices]
returned_route = swap_mutation(test_route, mutation_rate=1.0)
after  = [v.id for v in returned_route.vertices]

assert returned_route is test_route, "swap_mutation must return the same Route object"
ok("Returns the same Route object (in-place swap)")

diffs = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
assert len(diffs) == 2, f"Expected exactly 2 changed positions, got {len(diffs)}"
ok(f"Exactly 2 positions changed: indices {diffs}")
print(f"  Before: {before}")
print(f"  After:  {after}")

assert len(set(returned_route.vertices)) == 10, "Mutation broke route integrity"
ok("Route still contains all 10 unique locations after swap")

# Never mutate (rate = 0.0)
before2 = [v.id for v in test_route.vertices]
swap_mutation(test_route, mutation_rate=0.0)
after2  = [v.id for v in test_route.vertices]
assert before2 == after2, "mutation_rate=0.0 must never change the route"
ok("mutation_rate=0.0 -> route unchanged")

# Probabilistic check over many calls
random.seed(99)
mutated_count = 0
trials = 1000
rate   = 0.3
sample_route = Route(locations[:])
for _ in range(trials):
    before_ids = [v.id for v in sample_route.vertices]
    swap_mutation(sample_route, mutation_rate=rate)
    after_ids  = [v.id for v in sample_route.vertices]
    if before_ids != after_ids:
        mutated_count += 1
    # reset
    random.shuffle(sample_route.vertices)

ratio = mutated_count / trials
assert 0.20 <= ratio <= 0.40, \
    f"Expected ~30 % mutation rate, got {ratio:.1%}"
ok(f"Probabilistic gate correct: {ratio:.1%} mutations over {trials} trials (expected ~30 %)")


# ══ Summary ═══════════════════════════════════════════════════════════════════

print()
print("______________________________________________________________")
print("  All Member 2 tests passed.")
print("______________________________________________________________")
