import openml
import pprint

# Example dataset IDs — feel free to modify or expand this list
DATASET_IDS = [61, 40945, 45104, 42165]  # Iris, MNIST, Adult, Titanic

def dump_dataset_info(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_all_files=False)
        print(f"\n=== Dataset ID {dataset_id}: {dataset.name} ===\n")

        # Print all top-level attributes of the dataset object
        attrs = dataset.__dict__
        pprint.pprint(attrs)

        print("\n=== Qualities (automatically computed metadata) ===\n")
        pprint.pprint(dataset.qualities)

        print("\n=== Features ===\n")


        try:
            features = dataset.retrieve_class_labels()
            pprint.pprint(features)
        except Exception:
            print("No class labels or features available.")



    except Exception as e:
        print(f"Error fetching dataset {dataset_id}: {e}")

def main():
    for did in DATASET_IDS:
        dump_dataset_info(did)

if __name__ == "__main__":
    main()
