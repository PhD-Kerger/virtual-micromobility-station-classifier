import argparse
import json
import sys
import os
from pathlib import Path
from src.train_stations import StationTrainer
from src.classify_stations import StationClassifier
from src.logger import Logger
from src.pipeline_helpers import extract_network_data, classify_network_type


def main():
    parser = argparse.ArgumentParser(description="Virtual Station Trainer and Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train virtual stations for a city"
    )
    train_parser.add_argument("--city", required=True, help="Name of the city")
    train_parser.add_argument(
        "--locations",
        help="JSON string of locations: '[[lat, lng], ...]'",
    )
    train_parser.add_argument(
        "--locations-file",
        help="Path to a JSON file containing locations: [[lat, lng], ...]",
    )
    train_parser.add_argument(
        "--existing-stations-file",
        help="Path to a JSON file containing existing stations: [{'latitude': lat, 'longitude': lng, 'station_name': name, 'station_id': id}, ...]",
    )
    train_parser.add_argument(
        "--bandwidth", type=float, default=0.004, help="Clustering bandwidth"
    )
    train_parser.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Max files to process for data extraction (used for identifying new virtual stations)",
    )

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify locations into virtual stations"
    )
    classify_parser.add_argument("--city", required=True, help="Name of the city")
    classify_parser.add_argument(
        "--locations",
        help="JSON string of locations: '[[lat, lng], ...]'",
    )
    classify_parser.add_argument(
        "--locations-file",
        help="Path to a JSON file containing locations: [[lat, lng], ...]",
    )
    classify_parser.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Max files to process for data extraction",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Get information about a network (e.g., classification)"
    )
    info_parser.add_argument("--city", required=True, help="Name of the city/network")
    info_parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Max files to process for classification",
    )

    args = parser.parse_args()

    # Logger for main execution
    logger = Logger.get_logger(
        name="Main",
        log_file_path=Path("logs") / "logs.log",
    )

    def load_locations(locations_str, locations_file, city, mode, max_files=None):
        if locations_str:
            return json.loads(locations_str)
        if locations_file:
            with open(locations_file, "r") as f:
                return json.load(f)

        # Fallback to automated extraction if neither is provided
        logger.info(
            f"Neither --locations nor --locations-file provided. Attempting automated extraction for {city}..."
        )
        data = extract_network_data(
            city, mode=mode, max_files=max_files or args.max_files
        )
        if data is None:
            raise ValueError(
                f"Could not extract data for {city} and no manual files provided."
            )

        if mode == "classify":
            return data
        else:  # train
            return data["locations"], data["existing_stations"]

    if args.command == "train":
        try:
            locations_data = load_locations(
                args.locations,
                args.locations_file,
                args.city,
                "train",
                max_files=args.max_files,
            )

            if isinstance(locations_data, tuple):
                locations_list, existing_stations = locations_data
            else:
                locations_list = locations_data
                existing_stations = None
                if args.existing_stations_file:
                    with open(args.existing_stations_file, "r") as f:
                        existing_stations = json.load(f)

            trainer = StationTrainer()
            trainer.train_city_stations(
                args.city,
                locations_list,
                existing_stations=existing_stations,
                bandwidth=args.bandwidth,
            )
        except Exception as e:
            logger.error(f"Error in training command: {e}")
            import traceback

            logger.error(traceback.format_exc())
            sys.exit(1)

    elif args.command == "classify":
        try:
            locations_list = load_locations(
                args.locations,
                args.locations_file,
                args.city,
                "classify",
                max_files=args.max_files,
            )
            classifier = StationClassifier()
            results = classifier.classify_locations(args.city, locations_list)
            if results:
                print(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error in classify command: {e}")
            sys.exit(1)

    elif args.command == "info":
        try:
            logger.info(f"Running automated network classification for {args.city}...")
            result = classify_network_type(args.city, max_files=args.max_files)
            if result:
                print(json.dumps(result, indent=2))

                is_station_based = result.get("network_classification") == "Station-based"

                if is_station_based:
                    logger.info(
                        f"{args.city} is Station-based. Running only training/plotting for existing stations."
                    )
                else:
                    logger.info(f"Proceeding with standard training for {args.city}...")

                logger.info(f"Proceeding with training for {args.city}...")
                data = extract_network_data(
                    args.city, mode="train", max_files=args.max_files
                )
                if data:
                    trainer = StationTrainer()
                    # If station-based, ignore free-floating points to prevent virtual station creation
                    training_locations = [] if is_station_based else data["locations"]

                    trainer.train_city_stations(
                        args.city,
                        training_locations,
                        existing_stations=data["existing_stations"],
                        bandwidth=0.004,
                    )

                    if not is_station_based:
                        logger.info(f"Proceeding with classification for {args.city}...")
                        classifier = StationClassifier()
                        classify_data = extract_network_data(
                            args.city, mode="classify", max_files=args.max_files
                        )
                        if classify_data:
                            classifier.classify_locations(args.city, classify_data)
                    else:
                        logger.info(f"Classification skipped for Station-based network.")

            else:
                print(f"No info found for city: {args.city}")
        except Exception as e:
            logger.error(f"Error in info command: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
