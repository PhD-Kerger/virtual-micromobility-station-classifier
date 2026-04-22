import os
import pandas as pd
import pyarrow.parquet as pq
import yaml
from shapely import wkb
import re
from tqdm import tqdm
from src.logger import Logger


def _decode_location_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return wkb.loads(bytes(value))
        if isinstance(value, str):
            candidate = value.strip()
            if candidate and all(c in "0123456789abcdefABCDEF" for c in candidate):
                return wkb.loads(bytes.fromhex(candidate))
    except Exception:
        return None
    return None


def extract_coords(point_str):
    if not isinstance(point_str, str) or not point_str.startswith("POINT"):
        return None, None
    try:
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", point_str)
        if len(coords) >= 2:
            return float(coords[0]), float(coords[1])
    except Exception:
        pass
    return None, None


def make_location_readable(df):
    if "location" not in df.columns:
        return df
    decoded = df["location"].apply(_decode_location_value)
    if not decoded.notna().any():
        return df
    df = df.copy()
    df["location_raw"] = df["location"]
    df["location"] = decoded.apply(lambda g: g.wkt if g is not None else None)
    df["longitude"] = decoded.apply(lambda g: g.x if g is not None else None)
    df["latitude"] = decoded.apply(lambda g: g.y if g is not None else None)
    return df


def get_data_paths():
    config_path = os.path.join(os.path.dirname(__file__), "..", "env.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("data_paths", {})


NETWORK_NAME_MAPPINGS = {
    # Add Mappings for known alternative network names to the canonical name
    # "City-Bike-Wien": "CityBike Wien",
}


def extract_network_data(target_network, mode="train", max_files=None):
    logger = Logger.get_logger("DataExtractor")
    paths = get_data_paths()

    if not os.path.exists(paths["availability"]):
        logger.error(f"Availability directory not found: {paths['availability']}")
        return None

    files = sorted(
        [
            os.path.join(paths["availability"], f)
            for f in os.listdir(paths["availability"])
            if f.endswith(".parquet")
        ]
    )

    df_list = []
    match_count = 0

    columns = ["network_name", "location_id"]
    if mode == "train":
        columns.extend(["station_name_id", "station_id"])

    pbar = tqdm(files, desc=f"Extracting {target_network} data")

    # Pre-map common alternative names to direct value for faster lookup
    reverse_mappings = {}
    for k, v in NETWORK_NAME_MAPPINGS.items():
        if v == target_network:
            reverse_mappings[k] = True

    # Possible names for this network in the dataset
    target_names = {target_network} | set(reverse_mappings.keys())

    for file_path in pbar:
        try:
            # use filters to skip row groups that don't contain our network
            # if the parquet files have statistics/dictionaries stored.
            df = pd.read_parquet(
                file_path,
                columns=columns,
                filters=[("network_name", "in", list(target_names))],
            )
        except Exception:
            # Fallback if filters fail or columns are missing
            try:
                df = pd.read_parquet(file_path, columns=columns)
            except Exception:
                df = pd.read_parquet(file_path)

            if "network_name" in df.columns:
                df["network_name"] = (
                    df["network_name"]
                    .map(NETWORK_NAME_MAPPINGS)
                    .fillna(df["network_name"])
                )
            df = df[df["network_name"] == target_network]

        if not df.empty:
            df_list.append(df)
            match_count += 1
            pbar.set_postfix({"matches": match_count})

        if match_count == max_files:
            break

    if not df_list:
        logger.warning(f"No data found for network {target_network}")
        return None

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates(
        subset=["location_id"]
    )

    # Join with metadata
    if mode == "train":
        if os.path.exists(paths["station_names"]):
            station_names = pd.read_parquet(paths["station_names"])
            df_all = df_all.merge(station_names, on="station_name_id", how="left")

    if os.path.exists(paths["geo"]):
        geo_info = pd.read_parquet(paths["geo"])
        df_all = df_all.merge(geo_info, on="location_id", how="left")

    if "location" in df_all.columns:
        df_all = make_location_readable(df_all)
        coords = df_all["location"].apply(extract_coords)
        df_all["latitude"] = coords.apply(
            lambda x: x[0] if x else df_all.get("latitude")
        )
        df_all["longitude"] = coords.apply(
            lambda x: x[1] if x else df_all.get("longitude")
        )

    df_all = df_all.dropna(subset=["latitude", "longitude"])

    # Export intermediate files as requested (mirroring utils scripts)
    city_dir = os.path.join("data", target_network.lower())
    os.makedirs(city_dir, exist_ok=True)

    if mode == "classify":
        locations_list = df_all[["latitude", "longitude"]].values.tolist()
        with open(os.path.join(city_dir, "locations_classify.txt"), "w") as f:
            import json

            json.dump(locations_list, f)
        return locations_list

    else:  # train

        def is_excluded(name):
            if not isinstance(name, str):
                return True
            low = name.lower()
            return low.startswith("recording-") or "test" in low or "demo" in low

        df_all = df_all[~df_all["station_name"].apply(is_excluded)]
        df_ff = df_all[df_all["station_name"].str.startswith("BIKE", na=False)]
        df_st = df_all[~df_all["station_name"].str.startswith("BIKE", na=False)]

        locations_train = df_ff[["latitude", "longitude"]].values.tolist()
        existing_stations = df_st[
            ["latitude", "longitude", "station_name", "station_id"]
        ].to_dict(orient="records")

        with open(os.path.join(city_dir, "locations_train.txt"), "w") as f:
            import json

            json.dump(locations_train, f)

        with open(os.path.join(city_dir, "existing_stations.json"), "w") as f:
            import json

            json.dump(existing_stations, f)

        return {"locations": locations_train, "existing_stations": existing_stations}


def classify_network_type(network_name, max_files=50):
    logger = Logger.get_logger("NetworkClassifier")
    paths = get_data_paths()

    if not os.path.exists(paths["availability"]):
        logger.error(f"Availability directory not found: {paths['availability']}")
        return None

    # This mirrors the logic from get_free_floating.py
    files = sorted(
        [
            os.path.join(paths["availability"], f)
            for f in os.listdir(paths["availability"])
            if f.endswith(".parquet")
        ]
    )

    df_list = []
    limit = 0
    for file_path in tqdm(files, desc=f"Classifying {network_name}"):
        try:
            df = pd.read_parquet(
                file_path,
                columns=["network_name", "station_id", "station_name_id", "n_vehicles"],
            )
            df = df[df["network_name"] == network_name]
            if not df.empty:
                df_list.append(df)
                limit += 1
            if limit >= max_files:
                break
        except Exception:
            continue

    if not df_list:
        return None

    df_all = pd.concat(df_list, ignore_index=True)

    if os.path.exists(paths["station_names"]):
        station_names = pd.read_parquet(paths["station_names"])
        df_all = df_all.merge(station_names, on="station_name_id", how="left")

    # Node-level aggregation
    node_stats = (
        df_all.groupby(["network_name", "station_id"])
        .agg(
            name=("station_name", "first"),
            max_bikes=("n_vehicles", "max"),
            total_obs=("station_id", "count"),
        )
        .reset_index()
    )

    net_metadata = (
        node_stats.groupby("network_name")["total_obs"].max().rename("max_net_obs")
    )
    node_stats = node_stats.merge(net_metadata, on="network_name")

    node_stats["is_bike_name"] = node_stats["name"].str.contains(
        "^BIKE [0-9]+$", case=False, na=False, regex=True
    )
    node_stats["is_station"] = (
        ~node_stats["is_bike_name"]
        | (node_stats["max_bikes"] > 1)
        | (node_stats["total_obs"] > 0.75 * node_stats["max_net_obs"])
    )

    res_list = []
    for net, group in node_stats.groupby("network_name"):
        station_count = group["is_station"].sum()
        station_vol = group[group["is_station"]]["total_obs"].sum()
        total_vol = group["total_obs"].sum()
        station_ratio = station_vol / total_vol

        classification = "Free-floating"
        if station_count > 0:
            classification = "Station-based" if station_ratio >= 0.70 else "Mixed"

        res_list.append(
            {
                "network_name": net,
                "station_volume_ratio": round(float(station_ratio), 4),
                "network_classification": classification,
                "station_count": int(station_count),
                "total_nodes": int(len(group)),
            }
        )

    return res_list[0] if res_list else None
