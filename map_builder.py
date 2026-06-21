# map_builder.py
import folium
from folium.plugins import HeatMap

def build_heatmap(df, value_col, title, color_gradient=None):
    if df.empty:
        center = [12.9716, 77.5946]  # Bengaluru fallback
    else:
        center = [df['latitude'].mean(), df['longitude'].mean()]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
    heat_data = df[['latitude', 'longitude', value_col]].dropna().values.tolist()

    HeatMap(heat_data, radius=25, blur=15, max_zoom=13,
            gradient=color_gradient or {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
            ).add_to(m)

    # Mark top 5 hotspots with labeled pins
    top5 = df.sort_values(value_col, ascending=False).head(5)
    for i, row in enumerate(top5.itertuples(), start=1):
        folium.Marker(
            location=[row.latitude, row.longitude],
            popup=f"Rank #{i} | Zone {row.zone_dbscan} | Count: {getattr(row, value_col):.1f}",
            icon=folium.Icon(color='red', icon='exclamation-sign')
        ).add_to(m)

    title_html = f'<h3 style="text-align:center;font-family:Arial;">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    return m._repr_html_()