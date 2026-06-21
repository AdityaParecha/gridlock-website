# build_zone_centroids.py
import pandas as pd

df = pd.read_csv(r"C:\coding\GRIDLOCK-PHASE2\Problem-Statement-1\data\processed\zoned_data.csv")
centroids = df.groupby('zone_dbscan')[['latitude', 'longitude']].mean().reset_index()
centroids = centroids[centroids['zone_dbscan'] != '-1']  # drop DBSCAN noise label if present
centroids.to_csv("zone_centroids.csv", index=False)
print(f"Saved {len(centroids)} zone centroids.")