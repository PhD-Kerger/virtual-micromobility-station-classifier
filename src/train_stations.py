import os
import pickle
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import MeanShift
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPoint
from scipy.spatial import KDTree
from pathlib import Path
from src.logger import Logger
from scipy.spatial import ConvexHull


class StationTrainer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def voronoi_polygons(self, voronoi, bbox):
        """
        Generate finite Voronoi polygons clipped to a bounding box.
        """
        points = voronoi.points
        center = points.mean(axis=0)
        radius = np.max(np.abs(points - center)) * 100

        far_points = [
            center + radius * np.array([1, 1]),
            center + radius * np.array([1, -1]),
            center + radius * np.array([-1, 1]),
            center + radius * np.array([-1, -1]),
            center + radius * np.array([0, 1]),
            center + radius * np.array([0, -1]),
            center + radius * np.array([1, 0]),
            center + radius * np.array([-1, 0]),
        ]

        pts_extended = np.append(points, far_points, axis=0)
        vor_extended = Voronoi(pts_extended)

        valid_regions = []
        for i in range(len(points)):
            region_idx = vor_extended.point_region[i]
            region_indices = vor_extended.regions[region_idx]
            if -1 not in region_indices:
                poly = Polygon([vor_extended.vertices[j] for j in region_indices])
                clipped_poly = poly.intersection(bbox)
                if not clipped_poly.is_empty:
                    valid_regions.append(clipped_poly)

        return valid_regions

    def train_city_stations(
        self, city_name, locations, existing_stations=None, bandwidth=0.002
    ):
        """
        locations: list of [lat, lng]
        existing_stations: list of {'latitude': lat, 'longitude': lng, 'station_name': name, 'station_id': id}
        """
        self.logger.info(f"--- Training Virtual Stations for {city_name} ---")

        # Create city-specific directory
        city_dir = os.path.join(self.data_dir, city_name.lower())
        os.makedirs(city_dir, exist_ok=True)

        # 0. Data Prep: Filter and Outlier Detection
        df_locs = pd.DataFrame(locations, columns=["latitude", "longitude"])

        # Coordinates extraction for checking (Matches notebook Cell 8 logic)
        lats = df_locs["latitude"].dropna()
        lons = df_locs["longitude"].dropna()

        if not lats.empty:
            # Use Percentiles/IQR to find global spatial outliers (Cell 8) using only free-floating points
            q1_lat, q3_lat = lats.quantile(0.25), lats.quantile(0.75)
            iqr_lat = q3_lat - q1_lat

            q1_lon, q3_lon = lons.quantile(0.25), lons.quantile(0.75)
            iqr_lon = q3_lon - q1_lon

            # Define bounds (tightened to city area)
            lat_min, lat_max = q1_lat - 2 * iqr_lat, q3_lat + 2 * iqr_lat
            lon_min, lon_max = q1_lon - 2 * iqr_lon, q3_lon + 2 * iqr_lon

            initial_count = len(df_locs)
            df_locs = df_locs[
                (df_locs["latitude"] >= lat_min)
                & (df_locs["latitude"] <= lat_max)
                & (df_locs["longitude"] >= lon_min)
                & (df_locs["longitude"] <= lon_max)
            ]
            self.logger.info(
                f"Global Outlier Filter: Removed {initial_count - len(df_locs)} outliers from free-floating points."
            )

        # Drop duplicates based on location and ensure NO NaNs remain
        df_locs = df_locs.dropna(subset=["longitude", "latitude"]).drop_duplicates()

        # Convert to [longitude, latitude] for clustering (Matches notebook Cell 9 exactly)
        coords = df_locs[["longitude", "latitude"]].values

        # 1. MeanShift Clustering
        # If we have existing stations, we want to see if we need new virtual stations
        # where no existing station is "nearby".
        if len(coords) > 0:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=2)
            clusters = ms.fit_predict(coords)

            # Manual noise filtering (Matches notebook Cell 9 logic)
            df_locs["cluster_meanshift"] = clusters
            counts = df_locs["cluster_meanshift"].value_counts()
            single_point_clusters = counts[counts < 2].index
            df_locs.loc[
                df_locs["cluster_meanshift"].isin(single_point_clusters),
                "cluster_meanshift",
            ] = -1

            # Filter valid centers (those that are not marked as noise)
            all_centers = ms.cluster_centers_
            valid_cluster_indices = [
                i for i in range(len(all_centers)) if i not in single_point_clusters
            ]
            filtered_centers = all_centers[valid_cluster_indices]
        else:
            self.logger.info(
                "No free-floating locations provided. Skipping MeanShift clustering."
            )
            filtered_centers = []

        # Mixed Network Logic:
        # If existing_stations are provided, we use them as base stations.
        # We only add virtual stations if they are NOT near any existing station.
        final_hubs = []
        hub_names = []
        hub_ids = []

        if existing_stations:
            # existing_stations is list of {'latitude': lat, 'longitude': lng, 'station_name': name, 'station_id': id}
            for s in existing_stations:
                final_hubs.append([s["longitude"], s["latitude"]])
                hub_names.append(s["station_name"])
                hub_ids.append(s["station_id"])

            existing_coords = np.array([[s[0], s[1]] for s in final_hubs])
            existing_tree = KDTree(existing_coords)

            virtual_count = 1
            for center in filtered_centers:
                # Check if this center is "nearby" any existing station
                # Use bandwidth as distance threshold for "nearby"
                dist, idx = existing_tree.query(center)
                if dist > bandwidth:
                    final_hubs.append(center.tolist())
                    vst_name = f"VST-{city_name.upper()}-{virtual_count}"
                    hub_names.append(vst_name)
                    hub_ids.append(f"vst_{virtual_count}")
                    virtual_count += 1
        else:
            for i, center in enumerate(filtered_centers):
                final_hubs.append(center.tolist())
                vst_name = f"VST-{city_name.upper()}-{i+1}"
                hub_names.append(vst_name)
                hub_ids.append(f"vst_{i+1}")

        final_hubs = np.array(final_hubs)

        # Ensure final_hubs does not contain NaNs before spatial operations
        if len(final_hubs) > 0:
            nan_mask = np.isnan(final_hubs).any(axis=1)
            if nan_mask.any():
                self.logger.warning(
                    f"Removing {nan_mask.sum()} hubs with NaN coordinates."
                )
                final_hubs = final_hubs[~nan_mask]
                # Also need to filter hub_names and hub_ids to keep them in sync
                hub_names = [n for i, n in enumerate(hub_names) if not nan_mask[i]]
                hub_ids = [idx for i, idx in enumerate(hub_ids) if not nan_mask[i]]

        n_hubs = len(final_hubs)
        self.logger.info(
            f"Network established with {n_hubs} hubs ({len(existing_stations) if existing_stations else 0} existing, {n_hubs - (len(existing_stations) if existing_stations else 0)} virtual)."
        )

        # 2. Create Map (Voronoi and Points)
        if n_hubs >= 3:
            # Create a boundary that includes ALL hubs (especially peripheral existing stations)
            all_hubs_geom = MultiPoint(final_hubs)
            boundary = all_hubs_geom.convex_hull.buffer(
                0.02
            )  # Increased buffer to ensure outer stations are covered

            vor = Voronoi(final_hubs)
            polygons = self.voronoi_polygons(vor, boundary)

            avg_lat = np.mean(final_hubs[:, 1]) if n_hubs > 0 else 0
            avg_lon = np.mean(final_hubs[:, 0]) if n_hubs > 0 else 0
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

            # Matches COLORS list from notebook Cell 11/15
            COLORS_NOTEBOOK = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "darkred",
                "lightred",
                "beige",
                "darkblue",
                "darkgreen",
                "cadetblue",
                "darkpurple",
                "white",
                "pink",
                "lightblue",
                "lightgreen",
                "gray",
                "black",
                "lightgray",
            ]

            # Add groups for polygons as well so they are bundled with the markers
            station_poly_group = folium.FeatureGroup(name="Physical Stations")
            vst_poly_group = folium.FeatureGroup(name="Virtual Station Hubs")

            for i, poly in enumerate(polygons):
                color = COLORS_NOTEBOOK[i % len(COLORS_NOTEBOOK)]
                is_virtual = str(hub_ids[i]).startswith("vst_")

                # Enhanced popup with styling matching the theming (from tkg-odlocations)
                popup_html = f"""
                <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-width: 250px; max-width: 300px;">
                    <h1 style="margin: 0 0 15px 0; color: #2c5aa0; font-size: 18px; border-bottom: 2px solid #4682b4; padding-bottom: 8px;">
                        {'📍' if not is_virtual else '☁️'} {hub_names[i]}
                    </h1>
                    
                    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #4682b4; margin-bottom: 10px;">
                        <h4 style="margin: 0 0 8px 0; color: #2c5aa0; font-size: 14px;">ℹ️ Hub Details</h4>
                        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                            <tr><td style="padding: 3px 0;"><b>Name:</b></td><td style="text-align: right;">{hub_names[i]}</td></tr>
                            <tr><td style="padding: 3px 0;"><b>ID:</b></td><td style="text-align: right;"><code>{hub_ids[i]}</code></td></tr>
                            <tr><td style="padding: 3px 0;"><b>Type:</b></td><td style="text-align: right;">
                                <span style="background-color: {'#2ecc71' if not is_virtual else '#9b59b6'}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px;">
                                    {'Physical' if not is_virtual else 'Virtual'}
                                </span>
                            </td></tr>
                            <tr><td style="padding: 3px 0;"><b>Coordinates:</b></td><td style="text-align: right; font-size: 10px;">{final_hubs[i][1]:.4f}, {final_hubs[i][0]:.4f}</td></tr>
                        </table>
                    </div>
                </div>
                """

                folium.GeoJson(
                    poly,
                    style_function=lambda x, c=color: {
                        "fillColor": c,
                        "color": "white",
                        "weight": 2,
                        "fillOpacity": 0.4,
                    },
                    highlight_function=lambda x: {
                        "fillOpacity": 0.7,
                        "weight": 3,
                        "color": "white",
                    },
                    tooltip=f"{hub_names[i]}",
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(vst_poly_group if is_virtual else station_poly_group)

            station_poly_group.add_to(m)
            vst_poly_group.add_to(m)

            # Add markers to the same groups so toggling polygons also toggles markers
            for i, center in enumerate(final_hubs):
                if np.isnan(center).any():
                    self.logger.warning(
                        f"Skipping marker for station {hub_names[i]} due to NaN coordinates: {center}"
                    )
                    continue

                is_virtual = str(hub_ids[i]).startswith("vst_")

                # Use a small CircleMarker instead of a standard Marker for a cleaner look
                # Standard Markers can be too bulky on dense maps
                folium.CircleMarker(
                    location=[center[1], center[0]],
                    radius=4 if not is_virtual else 3,
                    color="white",
                    weight=1,
                    fill=True,
                    fill_color="cadetblue" if not is_virtual else "purple",
                    fill_opacity=1,
                    tooltip=f"{hub_names[i]}",
                    popup=folium.Popup(popup_html, max_width=400),
                ).add_to(vst_poly_group if is_virtual else station_poly_group)

            folium.LayerControl().add_to(m)

            map_path = os.path.join(city_dir, f"voronoi.html")
            m.save(map_path)
            self.logger.info(f"Map saved to {map_path}")

        # 3. Export KDTree and Convex Hull
        hull = ConvexHull(final_hubs)

        tree = KDTree(final_hubs)

        export_path = os.path.join(city_dir, f"kdtree.pkl")

        with open(export_path, "wb") as f:
            pickle.dump(
                {
                    "tree": tree,
                    "hubs": final_hubs,
                    "hub_names": hub_names,
                    "hub_ids": hub_ids,
                    "hull_vertices": final_hubs[hull.vertices],
                },
                f,
            )

        self.logger.info(f"System exported to {export_path}")
