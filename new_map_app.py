import streamlit as st
import pandas as pd
import numpy as np
import json
import pydeck as pdk
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

st.set_page_config(page_title="Park & Hub Routing Engine", layout="wide")
st.title("Park & Hub + Path Routing")

#load route data
@st.cache_data
def load_route_keys():
    with open('deployed_routes.json', 'r') as file: 
        return json.load(file)

all_route_data = load_route_keys()

# shift setup
st.sidebar.header("Shift Setup")
route_dictionary = {
    "CX88 (Green Lake/Wallingford)": "RouteID_0e8122d6-7a61-453b-af3c-150ef0a43110",
    "CX89 (U-District North)": "RouteID_03babb1c-6f77-4481-bf80-80fcf1a69ab1"
}
selected_short_code = st.sidebar.selectbox("Select Today's Route:", options=list(route_dictionary.keys()))
target_route = route_dictionary[selected_short_code]

#cluster enginer
@st.cache_data
def load_and_cluster_data(selected_route):
    stops = all_route_data[selected_route]['stops']
    route_list = [{'Stop_ID': stop_id, 'Lat': data['lat'], 'Lng': data['lng']} 
                  for stop_id, data in stops.items() if data['type'] != 'Station']
    if not route_list: return pd.DataFrame()

    df = pd.DataFrame(route_list)
    epsilon_radians = (700.0 / 5280.0) / 3958.8
    coords = np.radians(df[['Lat', 'Lng']])
    dist_matrix = haversine_distances(coords, coords)
    
    clusterer = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=epsilon_radians)
    df['Hub_ID'] = clusterer.fit_predict(dist_matrix)
    
    hub_counts = df['Hub_ID'].value_counts()
    valid_hubs = hub_counts[hub_counts >= 3].index
    df['Hub_ID'] = df['Hub_ID'].apply(lambda x: x if x in valid_hubs else -1)
    return df

df = load_and_cluster_data(target_route)

#path finder using nearest neighbor heuristic for TSP
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8 
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def solve_tsp(points_df):
    if len(points_df) < 2: return points_df
    
    points = points_df.to_dict('records')
    unvisited = points[1:]
    current_point = points[0]
    ordered_path = [current_point]
    
    while unvisited:
        # Find the closest next stop
        next_point = min(unvisited, key=lambda p: calculate_distance(current_point['Lat'], current_point['Lng'], p['Lat'], p['Lng']))
        ordered_path.append(next_point)
        unvisited.remove(next_point)
        current_point = next_point
        
    return pd.DataFrame(ordered_path)

# route selection
st.sidebar.divider()
st.sidebar.header("Turn-by-Turn Routing")
hub_options = ["Show All (No Paths)"] + [f"Hub {int(h)}" for h in df['Hub_ID'].unique() if h != -1] + ["Standalone Drive-Ups (-1)"]
selected_view = st.sidebar.radio("Select a Hub to Optimize:", hub_options)

#dashboard
def get_color(hub_id):
    if hub_id == -1: return [255, 0, 0, 200] 
    np.random.seed(hub_id)
    return [int(c) for c in np.random.randint(50, 255, 3)] + [200]

df['Color'] = df['Hub_ID'].apply(get_color)

layers = []

#Base Scatterplot
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["Lng", "Lat"],
    get_color="Color",
    get_radius=25, 
    pickable=True
))

#Optimized path
if selected_view != "Show All (No Paths)":
    target_id = -1 if "Standalone" in selected_view else int(selected_view.split(" ")[1])
    target_data = df[df['Hub_ID'] == target_id].copy()
    
    if len(target_data) > 1:
        optimized_df = solve_tsp(target_data)
        
        # Format the path for PyDeck
        path_data = pd.DataFrame({
            "path": [optimized_df[['Lng', 'Lat']].values.tolist()],
            "color": [[255, 255, 255, 255]] # White line for the route
        })
        
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color="color",
            width_scale=20,
            width_min_pixels=3,
            get_width=1
        ))

view_state = pdk.ViewState(latitude=df['Lat'].mean(), longitude=df['Lng'].mean(), zoom=14, pitch=0)
st.pydeck_chart(pdk.Deck(map_style="road", initial_view_state=view_state, layers=layers, tooltip={"text": "Stop ID: {Stop_ID}\nHub ID: {Hub_ID}"}))