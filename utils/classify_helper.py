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
        print("Usage: python classify_helper.py <CityName>")
        sys.exit(1)

    target_network = sys.argv[1]

    availability_folder_path = (
        "/path/to/availability/folder/"  # Update this path as needed
    )
    geo_information = pq.read_table("/path/to/geo_information.parquet").to_pandas()
    df_all = pd.DataFrame()

    # get all files sorted
    files = sorted(
        [
            os.path.join(availability_folder_path, f)
            for f in os.listdir(availability_folder_path)
            if f.endswith(".parquet")
        ]
    )

    df_list = []
    network_name_mappings = {
        "AW-bike": "AW-bike (VRM)",
        "AMB": "Barcelona",
        "BILBAO": "Bilbaobizi",
        "Bilbao": "Bilbaobizi",
        "Amprion": "Brauweiler / Rommerskirchen",
        "Brauweiler": "Brauweiler / Rommerskirchen",
        "Chorzów": "Chorzów (GZM)",
        "Agios Theodoros": "Fasoula",
        "GETXO": "Getxobizi",
        "Getxo": "Getxobizi",
        "Gothenborg Cargo": "Göteborg",
        "Halle": "Halle (Saale)",
        "Hochdorf": "Hochdorf / Seetal",
        "Katowice": "Katowice (GZM)",
        "Famagusta": "Larnaca",
        "Hlučín": "Ludgeřovice",
        "MV-Rad": "MV-Rad",
        "Oberdorf": "Oberdorf-Dallenwil",
        "Benešov": "OlbramoviceVotice",
        "Most": "OlbramoviceVotice",
        "Motovun": "Općina Motovun",
        "Kopřivnice": "Ostrava",
        "Pepingen": "Pepingen",
        "Prishtina": "Prishtinë",
        "Roosdaal": "Roosdaal",
        "Praha 22": "Rudná",
        "Rothenburg": "Ruswil",
        "Barakaldo": "Santurtzi",
        "Berango": "Santurtzi",
        "Bilbao": "Santurtzi",
        "Erandio": "Santurtzi",
        "Getxo": "Santurtzi",
        "Leioa": "Santurtzi",
        "Portugalete": "Santurtzi",
        "Sint-Pieters-Leeuw": "Sint-Pieters-Leeuw",
        "Grafschaft-Ringen": "Sinzig",
        "Remagen": "Sinzig",
        "Sosnowiec": "Sosnowiec (GZM)",
        "Split": "Split",
        "St. Pölten": "St.Pölten",
        "Büron": "Triengen-Büron",
        "Tulln": "Tulln an der Donau",
        "Tychy": "Tychy (GZM)",
        "Linou": "Vizakia",
        "Općina Vodnjan": "Vodnjan",
        "Wiener Neustadt": "WienerNeustadt",
        "Wisznia (WRM)": "Wisznia(WRM)",
        "Poličná": "Zašová",
        "Cyprus": "vyzakia",
        "Vizakia": "vyzakia",
        "Łódź (RL)": "Łódź - Rowerowe Łódzkie",
    }
    # Optimization: read files and filter for network_name directly.
    # No match_count limit here in classify_helper, but reading all columns is slow.
    pbar = tqdm(files, desc="Reading Parquet files")
    for file_path in pbar:
        # Optimization: read only necessary columns
        try:
            df = pd.read_parquet(file_path, columns=["network_name", "location_id"])
        except Exception:
            df = pd.read_parquet(file_path)

        if "network_name" in df.columns:
            df["network_name"] = (
                df["network_name"].map(network_name_mappings).fillna(df["network_name"])
            )

        df = df[df["network_name"] == target_network]

        if not df.empty:
            df_list.append(df)
            pbar.set_postfix({"matches": len(df_list)})

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True).drop_duplicates(
            subset=["location_id"]
        )
    else:
        df_all = pd.DataFrame()

    print(f"\nTotal unique locations for network '{target_network}': {len(df_all)}")

    # join with geo_information on location_id
    if not df_all.empty:
        if "location" in df_all.columns:
            df_all = make_location_readable(df_all)

        df_all = df_all.merge(geo_information, on="location_id", how="left")

        if "location" in df_all.columns and "latitude" not in df_all.columns:
            df_all = make_location_readable(df_all)

        # Assign cleaning coords
        coords = df_all["location"].apply(extract_coords)
        df_all["latitude"] = coords.apply(lambda x: x[0] if x else None)
        df_all["longitude"] = coords.apply(lambda x: x[1] if x else None)

        # return coords to be used like: python main.py train --city "Berlin" --locations "[[52.52, 13.40], [52.53, 13.41]]" to a file
        locations_list = df_all.dropna(subset=["latitude", "longitude"])[
            ["latitude", "longitude"]
        ].values.tolist()
        locations_json = json.dumps(locations_list)

    if not locations_list:
        print(f"No valid locations found for network '{target_network}'.")
    else:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        if not os.path.exists(f"./data/{target_network.lower()}"):
            os.makedirs(f"./data/{target_network.lower()}")
        with open(f"./data/{target_network.lower()}/locations_classify.txt", "w") as f:
            f.write(locations_json)

    print(f"\nLocations encoded successfully. Saved to 'locations_classify.txt'.")
