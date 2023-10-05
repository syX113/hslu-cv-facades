#!/usr/bin/env python

# Import common needed libraries 
import os
import cv2
import random
import numpy as np
import statistics

# CUDA & Torch setup
import torch
import detectron2
from GPUtil import showUtilization as gpu_usage
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Import additional detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from utils import format_predictions, compute_repeat_factors
# Custom COCO dataset function for detectron2
from detectron2.data.datasets import register_coco_instances

# Sampler to handle class imbalance
from detectron2.data import DatasetCatalog, MetadataCatalog


# Directory on server to images & COCO annotations
ROOT_DIR = '../../data/unzipped/facade-original-coco-segmentation/'

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


def setup(output_directory, sampling=False):

    # Register train & validation in COCO format
    register_coco_instances("facade_train_city", {}, ROOT_DIR + "train/_annotations.coco.json", ROOT_DIR + "train")
    register_coco_instances("facade_valid_city", {}, ROOT_DIR + "valid/_annotations.coco.json", ROOT_DIR + "valid")
    # Test dataset used later for the model evaluation but not used during training run (Hold-out strategy)
    register_coco_instances("facade_test_city", {}, ROOT_DIR + "test/_annotations.coco.json", ROOT_DIR + "test")
        
    # Define model configuration
    # Reference CFG parameters: https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" # Set GPU or CPU devices, e.g. "cuda:3" to use specific GPU
    cfg.OUTPUT_DIR = output_directory
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("facade_train_city",)
    cfg.DATASETS.TEST = ("facade_valid_city",)
    # cfg.TEST.EVAL_PERIOD = 100 # Done via Hooks (Iterations until validation is performed)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = [] # Learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 + 1  # Number of classes + background
    
    if sampling:
        repeat_factors = class_sampling('facade_train_city')
        # Sampling based on the computed class distributions
        cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        cfg.DATALOADER.REPEAT_THRESHOLD = statistics.fmean(repeat_factors)

    
    # Create the output directory if not exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def log_predictions(predictor, experiment, dataset_name="facade_test_city"):
    """Log Model Predictions to Comet for analysis.

    Args:
        predictor (DefaultPredictor): Predictor Object for Detectron Model
        experiment (comet_ml.Experiment): Comet Experiment Object
        dataset_name (str): Name of the registered dataset to use. Default is "facade_test_city".
    """
    # Retrieve the dataset dictionary using DatasetCatalog
    dataset_dicts = DatasetCatalog.get(dataset_name)

    predictions_data = {}
    for d in random.sample(dataset_dicts, 3):
        file_name = str(d["file_name"])

        im = cv2.imread(file_name)
        annotations = d["annotations"]

        outputs = predictor(
            im
        )  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        formatted_data = format_predictions(outputs, annotations)
        predictions_data[file_name] = formatted_data
        experiment.log_image(file_name, name=file_name)

    experiment.log_asset_data(predictions_data)