import minari

datasets = minari.list_remote_datasets()  # Get list of available remote datasets
dataset_ids =  datasets.keys()  # Extract dataset IDs
dataset_names = datasets.values()  # Extract dataset names

with open('minari_dataset_names.txt', 'w') as file:
    for id in dataset_ids:
        file.write(f"{id}\n")  # Write each name on a new line

print(f"Saved {len(dataset_names)} dataset names to minari_dataset_names.txt")