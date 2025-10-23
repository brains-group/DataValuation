import subprocess
import json
import os
import openml
import pprint

# === CONFIG ===
KAGGLE_DATASET = "hojjatk/mnist-dataset"
OPENML_IDS = [41063, 554]  # 41063: MNIST (OpenML), 554: Original MNIST
TEMP_DIR = "kaggle_metadata_temp"
OUTPUT_FILE = "mnist_combined_metadata.json"

os.makedirs(TEMP_DIR, exist_ok=True)

# === HELPERS ===
def get_kaggle_metadata(dataset_slug):
    """
    Downloads and parses Kaggle dataset metadata JSON properly as a nested object.
    """
    try:
        subprocess.run(
            ["kaggle", "datasets", "metadata", dataset_slug, "-p", TEMP_DIR],
            check=True, capture_output=True
        )
        metadata_file = os.path.join(TEMP_DIR, "dataset-metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                try:
                    # If Kaggle metadata is valid JSON, load normally
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Some Kaggle outputs wrap JSON quotes (stringified JSON)
                    fixed = content.strip('"').replace('\\"', '"')
                    data = json.loads(fixed)
            os.remove(metadata_file)

            # Ensure it's stored as a proper nested dict (not a string)
            return {
                "source": "kaggle",
                "slug": dataset_slug,
                "metadata": data
            }

        else:
            print(f"No metadata file found for {dataset_slug}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch {dataset_slug}: {e}")
        return None


def clean_kaggle_metadata(metadata):
    """
    Cleans Kaggle metadata so it's a proper nested dictionary,
    even if Kaggle returned a JSON string instead of a JSON object.
    """
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)  # if it's valid JSON
        except json.JSONDecodeError:
            try:
                fixed = metadata.strip('"').replace('\\"', '"')
                return json.loads(fixed)
            except Exception:
                return metadata
    return metadata


def get_openml_metadata(dataset_id):
    """
    Retrieves all OpenML dataset metadata and qualities.
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        meta = {
            "source": "openml",
            "id": dataset_id,
            "name": dataset.name,
            "description": getattr(dataset, "description", None),
            "attributes": dataset.__dict__,
            "qualities": dataset.qualities,
        }
        return meta
    except Exception as e:
        print(f"Error fetching OpenML dataset {dataset_id}: {e}")
        return None


# === MAIN ===
def main():
    combined_entry = {"kaggle": None, "openml": []}

    print(f"\nFetching Kaggle metadata for {KAGGLE_DATASET}...")
    kaggle_meta = get_kaggle_metadata(KAGGLE_DATASET)
    if kaggle_meta and "metadata" in kaggle_meta:
        kaggle_meta["metadata"] = clean_kaggle_metadata(kaggle_meta["metadata"])
    combined_entry["kaggle"] = kaggle_meta

    for did in OPENML_IDS:
        print(f"\nFetching OpenML metadata for dataset ID {did}...")
        openml_meta = get_openml_metadata(did)
        if openml_meta:
            combined_entry["openml"].append(openml_meta)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_entry, f, indent=4, ensure_ascii=False)

    print(f"\nCombined metadata saved to {OUTPUT_FILE}\n")
    pprint.pprint(combined_entry)


if __name__ == "__main__":
    main()