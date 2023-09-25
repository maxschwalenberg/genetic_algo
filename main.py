import numpy as np
import imageio
import os
import natsort

from code.utils import (
    create_gdf_from_names,
    create_distance_matrix,
)
from code.tsp import TSPGeneticAlgo


def create_gif(output_path: str):
    filenames = natsort.natsorted(os.listdir(output_path))
    print(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(output_path, filename)))
    imageio.mimsave(os.path.join(output_path, "animation.gif"), images)


def solve_tsp(distance_matrix, gdf, points, output_path: str, algorithm=None):
    if algorithm == "genetic_algo":
        solver = TSPGeneticAlgo(
            distance_matrix, points, gdf, 10, 50, 0.1, 0.05, output_path
        )
        solver.run_evolution()
    elif algorithm == "symmetric":
        # paul
        pass
    else:
        raise NotImplementedError(f"Not yet implemented {algorithm}")


def get_city_names():
    return [
        "magdeburg",
        "berlin",
        "rostock",
        "rom",
        "madrid",
        "dortmund",
        "münchen",
        "wien",
        "paris",
        "london",
        "prag",
    ]


def preprocess_cities(cities: list[str]):
    # preprocess cities to coords, ... etc
    gdf = create_gdf_from_names(cities)
    points = np.column_stack((np.array(gdf.lon.tolist()), np.array(gdf.lat.tolist())))

    distance_matrix = create_distance_matrix(points)
    return distance_matrix, points, gdf


def main():
    output_path = "output"
    cities = get_city_names()
    distance_matrix, points, gdf = preprocess_cities(cities)

    solve_tsp(distance_matrix, gdf, points, output_path, algorithm="genetic_algo")

    create_gif(output_path)


if __name__ == "__main__":
    main()
