import os
import pickle
from pathlib import Path
from src.logger import Logger
import pandas as pd
from matplotlib.path import Path as MplPath


class StationClassifier:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def _is_in_hull(self, point, hull_path):
        """
        Tests if a point is within a convex hull using Path.contains_points.
        point: [lng, lat]
        hull_path: matplotlib.path.Path object
        """
        return hull_path.contains_point(point)

    def classify_locations(self, city_name, locations):
        """
        locations: list of [lat, lng]
        Returns: list of dicts with virtual_station_id, virtual_station_lat, virtual_station_lng, and distance_m
        """
        import_path = os.path.join(self.data_dir, city_name.lower(), "kdtree.pkl")

        if not os.path.exists(import_path):
            self.logger.error(
                f"No station model found for city '{city_name}' at {import_path}"
            )
            return None

        # 1. Load the model
        try:
            with open(import_path, "rb") as f:
                data = pickle.load(f)

            tree = data["tree"]
            hubs = data.get("hubs")  # [lng, lat]
            hub_names = data.get("hub_names", [])
            hub_ids = data.get("hub_ids", [])
            hull_vertices = data.get("hull_vertices")

            # Prepare Path object for inclusion test if hull available
            hull_path = None
            if hull_vertices is not None:
                hull_path = MplPath(hull_vertices)

            # 2. Classify
            results = []
            for lat, lng in locations:
                point = [lng, lat]

                # Distance calculation for fuzzy hull buffer
                dist, station_idx = tree.query(point)
                dist_meters = dist * 111000  # Approx

                # Check if point is within convex hull
                in_hull = self._is_in_hull(point, hull_path) if hull_path else True
                
                # Outlier if outside hull AND further than 250m from nearest hub
                if not in_hull and dist_meters > 250:
                    results.append(
                        {
                            "lat": lat,
                            "lng": lng,
                            "station_id": -1,
                            "station_name": None,
                            "station_lat": None,
                            "station_lng": None,
                            "distance_m": dist_meters,
                            "is_virtual": False,
                            "status": "outlier",
                        }
                    )
                    continue

                # Get site coordinates if hubs available
                s_lat, s_lng = (None, None)
                s_name = None
                s_id = None
                is_virtual = False

                if hubs is not None:
                    s_lng, s_lat = hubs[station_idx]
                    s_lat = round(s_lat, 3)
                    s_lng = round(s_lng, 3)

                if station_idx < len(hub_names):
                    s_name = hub_names[station_idx]
                    s_id = hub_ids[station_idx]
                    is_virtual = str(s_id).startswith("vst_")

                results.append(
                    {
                        "lat": lat,
                        "lng": lng,
                        "station_id": s_id,
                        "station_name": s_name,
                        "station_lat": s_lat,
                        "station_lng": s_lng,
                        "distance_m": round(dist_meters, 2),
                        "is_virtual": is_virtual,
                        "status": "valid",
                    }
                )

            # write results to csv
            df_results = pd.DataFrame(results)
            output_path = os.path.join(
                self.data_dir, city_name.lower(), "classified_locations.csv"
            )
            df_results.to_csv(output_path, index=False)
            self.logger.info(
                f"Classification completed for {city_name}. Results saved to {output_path}"
            )

            return results
        except Exception as e:
            self.logger.error(f"Error classifying locations for {city_name}: {e}")
            return None
