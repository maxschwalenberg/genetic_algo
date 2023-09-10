from geopy.geocoders import Nominatim
import geopandas as gpd
import pandas as pd

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


