import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pydeck as pdk
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

st.set_page_config(page_title="Park & Hub Map", layout="wide")
st.title("Phase 2: Park & Hub Routing Engine")
st.markdown("Algorithm: Agglomerative Clustering (Strict 350ft Walking Radius)")

#Load routes
@st.cache_data
def load_route_keys():
    with open('deployed_routes.json', 'r') as file: 
        route_data = json.load(file)
    return route_data

all_route_data = load_route_keys()

#userinput for route selection
st.sidebar.header("Shift Setup")

route_dictionary = {
    "CX88 (Green Lake/Wallingford)": "RouteID_0e8122d6-7a61-453b-af3c-150ef0a43110",
    "CX89 (U-District North)": "RouteID_03babb1c-6f77-4481-bf80-80fcf1a69ab1"
}

selected_short_code = st.sidebar.selectbox(
    "Select Today's Discord Route:", 
    options=list(route_dictionary.keys())
)

target_route = route_dictionary[selected_short_code]

#cluster engine with caching to speed up repeated runs on the same route selection
@st.cache_data
def load_and_cluster_data(selected_route):
    stops = all_route_data[selected_route]['stops']

    route_list = [{'Stop_ID': stop_id, 'Lat': data['lat'], 'Lng': data['lng']} 
                  for stop_id, data in stops.items() if data['type'] != 'Station']
    
    #if the route is empty
    if not route_list:
        return pd.DataFrame()

    df = pd.DataFrame(route_list)

    #The constraint
    epsilon_radians = (700.0 / 5280.0) / 3958.8
    coords = np.radians(df[['Lat', 'Lng']])
    dist_matrix = haversine_distances(coords, coords)
    
    clusterer = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=epsilon_radians)
    df['Hub_ID'] = clusterer.fit_predict(dist_matrix)
    
    #Filter out stops that are too small (1 or 2 packages)
    hub_counts = df['Hub_ID'].value_counts()
    valid_hubs = hub_counts[hub_counts >= 3].index
    df['Hub_ID'] = df['Hub_ID'].apply(lambda x: x if x in valid_hubs else -1)
    
    return df

#user selection to load and cluster data
df = load_and_cluster_data(target_route)

#color coding function for map visualization
def get_color(hub_id):
    if hub_id == -1:
        return [255, 0, 0, 200] 
    np.random.seed(hub_id)
    return [int(c) for c in np.random.randint(50, 255, 3)] + [200]

#render dashboard metrics and map visualization
if df.empty:
    st.error("No valid stop data found for this route.")
else:
    df['Color'] = df['Hub_ID'].apply(get_color)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Deliveries", len(df))
    col2.metric("Walking Hubs Formed", df[df['Hub_ID'] != -1]['Hub_ID'].nunique())
    col3.metric("Standalone Drive-Up Stops", len(df[df['Hub_ID'] == -1]))

    view_state = pdk.ViewState(
        latitude=df['Lat'].mean(),
        longitude=df['Lng'].mean(),
        zoom=14,
        pitch=0
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["Lng", "Lat"],
        get_color="Color",
        get_radius=20, 
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style="road",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Stop ID: {Stop_ID}\nHub ID: {Hub_ID}"}
    ))