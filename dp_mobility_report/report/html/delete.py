import json
import folium
import requests


url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
antarctic_ice_edge = f"{url}/antarctic_ice_edge.json"
antarctic_ice_shelf_topo = f"{url}/antarctic_ice_shelf_topo.json"

folium.folium._default_css.append(('leaflet_overloaded_css', 'https://your_url/your_css_file.css'))

m = folium.Map(
    location=[-59.1759, -11.6016],
    tiles=None,
    zoom_start=2,
)

tile_layer = folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png",
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    max_zoom=19,
    name='darkmatter',
    control=False,
    opacity=0.7
)
tile_layer.add_to(m)

folium.GeoJson(antarctic_ice_edge, name="geojson").add_to(m)

folium.TopoJson( 
    json.loads(requests.get(antarctic_ice_shelf_topo).text), 
    "objects.antarctic_ice_shelf", 
    name="topojson", 
).add_to(m) 

folium.LayerControl().add_to(m)
m.save('demo.html')