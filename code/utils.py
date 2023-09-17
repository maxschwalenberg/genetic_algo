import math

from geopy.geocoders import Nominatim
import geopandas as gpd
import pandas as pd
import numpy as np

def get_coordinates_from_name(city_string: str):
    geolocator = Nominatim(user_agent='myapplication')
    location = geolocator.geocode(city_string)

    if location is None:
        # no coordinates could be found
        return None
    else:
        lon, lat = (location.longitude, location.latitude)

    # TODO add logging

    return lon,lat


def create_gdf_from_names(city_names: list[str]):
    data = []
    for city in city_names:
        coords = get_coordinates_from_name(city)
        if coords is None:
            continue
        else:
            data.append([city, coords[0], coords[1]])
    
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['name', 'lon', 'lat'])

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )   

    return gdf


def get_basemap_params_from_gdf(gdf: gpd.GeoDataFrame, zoom_out: float):
    lons = gdf["lon"].tolist()
    lats = gdf["lat"].tolist()

    # longitude is west-east ... latitude north-south
    min_lon, max_lon = (min(lons), max(lons))
    min_lat, max_lat = (min(lats), max(lats))


    longitude_zoom_value = (max_lon - min_lon) * zoom_out
    latitude_zoom_value = (max_lat - min_lat) * zoom_out



    mid_point = (min_lon + (max_lon - min_lon) / 2, min_lat + (max_lat - min_lat) / 2)

    lower_left_corner = (min_lon - longitude_zoom_value / 2, min_lat - latitude_zoom_value / 2)
    upper_right_corner = (max_lon + longitude_zoom_value / 2, max_lat + latitude_zoom_value / 2)

    return lower_left_corner, upper_right_corner, mid_point


def distance_between_two_points(coord_1, coord_2):
    # haversine formula
    lon_1, lat_1, lon_2, lat_2 = (coord_1[0], coord_1[1], coord_2[0], coord_2[1])

    r = 6371 # km
    p = math.pi / 180

    a = 0.5 - math.cos((lat_2-lat_1)*p)/2 + math.cos(lat_1*p) * math.cos(lat_2*p) * (1-math.cos((lon_2-lon_1)*p))/2
    return 2 * r * math.asin(math.sqrt(a))


def create_distance_matrix(points):
    n_points = len(points)

    distance_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = distance_between_two_points(points[i], points[j])

            # matrix is symmetrical
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d

    return distance_matrix