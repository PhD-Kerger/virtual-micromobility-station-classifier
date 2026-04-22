import os
import sys
import pyarrow.parquet as pq
import pandas as pd
from shapely import wkb
import re
import json
from tqdm import tqdm


def read_parquet_file(file_path):
    try:
        table = pq.read_table(file_path)
        return table
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None


def get_column_names(table):
    if table:
        return table.column_names
    return []


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


def make_location_readable(df):
    if "location" not in df.columns:
        return df

    decoded = df["location"].apply(_decode_location_value)
    has_geometry = decoded.notna()

    if not has_geometry.any():
        return df

    # Preserve raw value and expose readable geometry + coordinates.
    df = df.copy()
    df["location_raw"] = df["location"]
    df["location"] = decoded.apply(lambda g: g.wkt if g is not None else None)
    df["longitude"] = decoded.apply(lambda g: g.x if g is not None else None)
    df["latitude"] = decoded.apply(lambda g: g.y if g is not None else None)

    return df


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_helper.py <CityName>")
        sys.exit(1)

    target_network = sys.argv[1]

    availability_folder_path = (
        "/path/to/availability/folder/"  # Update this path as needed
    )
    geo_information = pq.read_table("/path/to/geo_information.parquet").to_pandas()
    # User mentioned station names parquet file
    station_names = pq.read_table("/path/to/station_names.parquet").to_pandas()

    df_all = pd.DataFrame()

    # get all files sorted
    files = sorted(
        [
            os.path.join(availability_folder_path, f)
            for f in os.listdir(availability_folder_path)
            if f.endswith(".parquet")
        ]
    )

    # Use pandas to read all files at once and filter, much faster than custom read_table in a loop
    df_list = []
    match_count = 0
    network_name_mappings = {
        # Add Mappings for known alternative network names to the canonical name
        # "City-Bike-Wien": "CityBike Wien",
    }

    pbar = tqdm(files, desc="Reading Parquet files")
    for file_path in pbar:
        # Optimization: read only necessary columns
        try:
            df = pd.read_parquet(
                file_path,
                columns=[
                    "network_name",
                    "location_id",
                    "station_name_id",
                    "station_id",
                ],
            )
        except Exception:
            # Fallback if columns are different
            df = pd.read_parquet(file_path)

        if "network_name" in df.columns:
            df["network_name"] = (
                df["network_name"].map(network_name_mappings).fillna(df["network_name"])
            )

        df = df[df["network_name"] == target_network]

        if not df.empty:
            df_list.append(df)
            match_count += 1
            pbar.set_postfix({"matches": match_count})

        if match_count >= 30:
            print("Reached 30 matches. Stopping.")
            break

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True).drop_duplicates(
            subset=["location_id"]
        )
    else:
        df_all = pd.DataFrame()

    # join with geo_information on location_id
    if not df_all.empty:
        # Join with station names
        df_all = df_all.merge(station_names, on="station_name_id", how="left")

        # Join with geo_information
        df_all = df_all.merge(geo_information, on="location_id", how="left")

        # Process location data
        if "location" in df_all.columns:
            df_all = make_location_readable(df_all)

        # Assign cleaning coords
        coords = df_all["location"].apply(extract_coords)
        df_all["latitude"] = coords.apply(lambda x: x[0] if x else None)
        df_all["longitude"] = coords.apply(lambda x: x[1] if x else None)

        # Filter by station name rules
        # Free floating: "BIKE-" prefix
        # Stations: everything else
        # Exclude: "recording-", "test", "demo"

        def is_excluded(name):
            if not isinstance(name, str):
                return True
            lowered = name.lower()
            return (
                lowered.startswith("recording-")
                or "test" in lowered
                or "demo" in lowered
            )

        df_all = df_all[~df_all["station_name"].apply(is_excluded)]

        df_free_floating = df_all[
            df_all["station_name"].str.startswith("BIKE", na=False)
        ]
        df_stations = df_all[~df_all["station_name"].str.startswith("BIKE", na=False)]

        # Prepare outputs
        free_floating_coords = (
            df_free_floating[["latitude", "longitude"]].dropna().values.tolist()
        )

        stations_data = (
            df_stations[["latitude", "longitude", "station_name", "station_id"]]
            .dropna(subset=["latitude", "longitude"])
            .to_dict(orient="records")
        )

        # Save to files
        os.makedirs(f"data/{target_network.lower()}", exist_ok=True)

        with open(f"data/{target_network.lower()}/locations_train.txt", "w") as f:
            json.dump(free_floating_coords, f)

        with open(f"data/{target_network.lower()}/existing_stations.json", "w") as f:
            json.dump(stations_data, f)

        print(
            f"Saved {len(free_floating_coords)} free floating locations and {len(stations_data)} existing stations."
        )
