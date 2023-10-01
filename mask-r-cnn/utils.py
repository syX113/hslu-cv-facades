# Original source: https://github.com/comet-ml/comet-detectron/blob/main/utils.py
# Extended and adjusted to meet the requirements, e.g. repeat factor calculation

import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def format_predictions(outputs, annotations):
    """Format Dectectron Predictions so that they can be visualized using
    Comet Panels

    Args:
        outputs (list): List of formatted dicts
    """
    data = []
    prediction = outputs["instances"].to("cpu")

    predicted_boxes = prediction.pred_boxes.tensor.numpy().tolist()
    predicted_scores = prediction.scores.numpy().tolist()
    predicted_classes = prediction.pred_classes.numpy().tolist()

    for annotation in annotations:
        bbox = annotation["bbox"]

        # Convert from numpy.int64 to int, if needed
        x, y, x2, y2 = map(lambda x: x.item() if hasattr(x, 'item') else x, bbox)

        label = annotation["category_id"]
        data.append(
            {
                "label": f"ground_truth-{label}",
                "score": 100,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
            }
        )

    for predicted_box, predicted_score, predicted_class in zip(
        predicted_boxes, predicted_scores, predicted_classes
    ):
        x, y, x2, y2 = predicted_box
        data.append(
            {
                "label": predicted_class,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
                "score": predicted_score * 100,
            }
        )

    return data


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