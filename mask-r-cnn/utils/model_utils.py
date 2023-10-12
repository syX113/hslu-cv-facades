from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np

def compute_repeat_factors(dataset_name):
    data_dicts = DatasetCatalog.get(dataset_name)
    class_names = MetadataCatalog.get(dataset_name).thing_classes

    # Compute class frequencies
    class_counts = {classname: 0 for classname in class_names}
    for data_dict in data_dicts:
        for anno in data_dict["annotations"]:
            class_counts[class_names[anno["category_id"]]] += 1

    # Compute the repeat factors for each instance in the dataset.
    repeat_factors = []
    for data_dict in data_dicts:
        factor = 1.0  # Default factor
        for anno in data_dict["annotations"]:
            factor = max(factor, 1.0 / class_counts[class_names[anno["category_id"]]])
        repeat_factors.append(factor)

    return repeat_factors


def class_sampling(dataset_name):
    
    repeat_factors = compute_repeat_factors(dataset_name)
    
    # Print class distribution before sampling
    data_dicts = DatasetCatalog.get(dataset_name)
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    class_counts = {classname: 0 for classname in class_names}
    for data_dict in data_dicts:
        for anno in data_dict["annotations"]:
            class_counts[class_names[anno["category_id"]]] += 1
    total_instances = sum(class_counts.values())
    class_frequencies = {k: v / total_instances for k, v in class_counts.items()}
    print("Class Distribution (before sampling):", class_frequencies)

    # Print class distribution after sampling
    effective_counts = {k: v * np.mean(repeat_factors) for k, v in class_counts.items()}
    effective_distribution = {k: v / sum(effective_counts.values()) for k, v in effective_counts.items()}
    print("Class Distribution (after sampling):", effective_distribution)
    
    return repeat_factors