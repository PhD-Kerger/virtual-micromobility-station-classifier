import pandas as pd
import os
import argparse
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def classify_network_robust(df):
    """
    Robustly classifies a network as Station-based, Free-floating, or Mixed.

    Logic:
    - STATION-BASED: Activity is concentrated at persistent stations (Ratio >= 0.70).
    - FREE-FLOATING: Virtually no station presence (< 1% station Activity AND no significant stations).
    - MIXED: stations exist but a significant portion of activity is floating (Ratio 1%-70%).

    CRITICAL: If a network has any persistent stations (stations), it cannot be Free-floating.
    """
    if df.empty:
        return None

    # Node-level aggregation
    node_stats = (
        df.groupby(["network_name", "station_id"])
        .agg(
            name=("station_name", "first"),
            max_bikes=("n_vehicles", "max"),
            total_obs=("station_id", "count"),
        )
        .reset_index()
    )

    # Get maximum captures in this period per network for normalization
    net_metadata = (
        node_stats.groupby("network_name")["total_obs"].max().rename("max_net_obs")
    )
    node_stats = node_stats.merge(net_metadata, on="network_name")

    # Characterize stations
    node_stats["is_bike_name"] = node_stats["name"].str.contains(
        "^BIKE [0-9]+$", case=False, na=False, regex=True
    )

    # A location is a "station" if it's named, acts as a pool, or persists across many snapshots
    node_stats["is_station"] = (
        ~node_stats["is_bike_name"]
        | (node_stats["max_bikes"] > 1)
        | (node_stats["total_obs"] > 0.75 * node_stats["max_net_obs"])
    )

    results = []
    for net, group in node_stats.groupby("network_name"):
        station_count = group["is_station"].sum()
        station_vol = group[group["is_station"]]["total_obs"].sum()
        total_vol = group["total_obs"].sum()
        station_ratio = station_vol / total_vol

        if station_count > 0:
            if station_ratio >= 0.70:
                classification = "Station-based"
            else:
                classification = "Mixed"
        else:
            classification = "Free-floating"

        results.append(
            {
                "network_name": net,
                "station_volume_ratio": round(float(station_ratio), 4),
                "network_classification": classification,
                "station_count": int(station_count),
                "total_nodes": int(len(group)),
            }
        )

    return results


def run_classification_for_network(
    network_name, availability_dir, station_names_path, max_files=50
):
    """
    Helper function to load data and run classification for a single network.
    """
    # Load station names
    logger.info(f"Loading station names from {station_names_path}")
    try:
        station_names = pd.read_parquet(station_names_path)
    except Exception as e:
        logger.error(f"Failed to load station names: {e}")
        return {"error": f"Failed to load station names: {e}"}

    # Load availability data
    logger.info(f"Scanning availability directory: {availability_dir}")
    files = sorted(
        [
            os.path.join(availability_dir, f)
            for f in os.listdir(availability_dir)
            if f.endswith(".parquet")
        ]
    )[:max_files]

    logger.info(f"Processing {len(files)} parquet files for network: {network_name}")
    df_all = pd.DataFrame()
    for file_path in tqdm(files, desc="Reading Parquet files"):
        try:
            df = pd.read_parquet(
                file_path,
                columns=["network_name", "station_id", "station_name_id", "n_vehicles"],
            )
            df = df[df["network_name"] == network_name]
            if not df.empty:
                df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            logger.debug(f"Skipping {os.path.basename(file_path)}: {e}")
            continue

    if df_all.empty:
        logger.warning(f"No data found for network: {network_name}")
        return {
            "network_name": network_name,
            "error": "No data found for this network.",
        }

    logger.info(f"Total observations collected: {len(df_all)}")

    # Join with station names
    df_all = df_all.merge(station_names, on="station_name_id", how="left")

    # Classify
    logger.info(f"Classifying network: {network_name}")
    results = classify_network_robust(df_all)

    if results:
        res = results[0]
        logger.info(
            f"Classification complete: {res['network_classification']} (Ratio: {res['station_volume_ratio']})"
        )
        return res
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a bike network based on persistence logic."
    )
    parser.add_argument(
        "network", help="Name of the network to classify (e.g., 'Berlin')"
    )
    parser.add_argument(
        "--avail-dir",
        help="Path to availability parquet folder",
    )
    parser.add_argument(
        "--stations-path",
        help="Path to station_names.parquet",
    )

    args = parser.parse_args()

    result = run_classification_for_network(
        args.network, args.avail_dir, args.stations_path
    )
    print(json.dumps(result, indent=4))
