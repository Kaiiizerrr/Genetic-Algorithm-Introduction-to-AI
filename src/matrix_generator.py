import math
from pathlib import Path

import pandas as pd


class Location:
    def __init__(self, id, x, y, name=None):
        self.id = id
        self.x = x
        self.y = y
        self.name = name

    def __repr__(self):
        if self.name:
            return f"Location(id={self.id}, name={self.name!r}, x={self.x}, y={self.y})"
        return f"Location(id={self.id}, x={self.x}, y={self.y})"


def read_locations_from_excel(filename="CX Group 5.xlsx"):
    """Read locations from an Excel file and return a list of Location objects."""
    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path

    try:
        df = pd.read_excel(file_path, usecols=["City_ID", "X_Coordinate", "Y_Coordinate"])
        df["Location"] = None
    except ValueError:
        df = pd.read_excel(
            file_path,
            sheet_name="Coordinates",
            header=2,
            usecols="B:D",
            names=["Location", "X_Coordinate", "Y_Coordinate"],
        )
        df = df.dropna(subset=["X_Coordinate", "Y_Coordinate"]).reset_index(drop=True)
        df.insert(0, "City_ID", range(1, len(df) + 1))

    locations = []
    for record in df.to_dict(orient="records"):
        name = record.get("Location")
        if pd.isna(name):
            name = None

        location = Location(
            id=int(record["City_ID"]),
            x=float(record["X_Coordinate"]),
            y=float(record["Y_Coordinate"]),
            name=name,
        )
        locations.append(location)

    return locations


def build_distance_matrix(locations, size=10):
    """Build a size x size Euclidean distance matrix from the first `size` locations."""
    if len(locations) < size:
        raise ValueError(f"Expected at least {size} locations, found {len(locations)}")

    selected_locations = locations[:size]
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(i + 1, size):
            dx = selected_locations[j].x - selected_locations[i].x
            dy = selected_locations[j].y - selected_locations[i].y
            distance = math.sqrt(dx * dx + dy * dy)
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


def load_locations_and_matrix(filename="CX Group 5.xlsx", size=10):
    """Load locations from Excel and return both the Location list and the distance matrix."""
    locations = read_locations_from_excel(filename)
    matrix = build_distance_matrix(locations, size=size)
    return locations[:size], matrix


if __name__ == "__main__":
    locations, distance_matrix = load_locations_and_matrix()
    print("Loaded locations:")
    for loc in locations:
        print(loc)
    print("\nDistance matrix:")
    for row in distance_matrix:
        print(row)
