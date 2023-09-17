import numpy as np

from code.utils import create_gdf_from_names, get_basemap_params_from_gdf, create_distance_matrix

def solve_tsp(distance_matrix, algorithm = None):
    

    print(distance_matrix)



def get_city_names():
    return ["magdeburg", "berlin", "rostock", "kopenhagen", "madrid"]


def preprocess_cities(cities: list[str]):
    # preprocess cities to coords, ... etc
    gdf = create_gdf_from_names(cities)
    points = np.column_stack((np.array(gdf.lon.tolist()), np.array(gdf.lat.tolist())))


    distance_matrix = create_distance_matrix(points)
    return distance_matrix

def main():
    cities = get_city_names()
    distance_matrix = preprocess_cities(cities)
    solve_tsp(distance_matrix)

if __name__ == "__main__":
    main()