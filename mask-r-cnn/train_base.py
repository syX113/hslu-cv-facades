# Import common needed libraries 
import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# Setup & initialize Comet logging
with open('mask-r-cnn/comet_api_key.txt', 'r') as file:
    api_key = file.read()
os.environ['COMET_API_KEY'] = api_key
os.environ['COMET_PROJECT_NAME'] = 'hslu-computer-vision'
import comet_ml
from comet_trainer import CometDefaultTrainer
comet_ml.init()

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
from detectron2.engine import DefaultPredictor, hooks
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from utils import format_predictions, get_facade_dicts
# Custom COCO dataset function for detectron2
from detectron2.data.datasets import register_coco_instances


def clear_gpu_cache():

    print("Initial GPU Usage: \n")
    gpu_usage()
    print("GPU Usage after emptying the cache: \n")
    torch.cuda.empty_cache()
    gpu_usage()


def setup(output_directory):
    
    # Directory on server to images & COCO annotations
    root_dir = '../../data/unzipped/facade-original-coco-segmentation/'

    # Register train & validation in COCO format
    register_coco_instances("facade_train_city", {}, root_dir + "train/_annotations.coco.json", root_dir + "train")
    register_coco_instances("facade_valid_city", {}, root_dir + "valid/_annotations.coco.json", root_dir + "valid")
    
    # Test dataset used later for the model evaluation but not used during training run (Hold-out strategy)
    register_coco_instances("facade_test_city", {}, root_dir + "test/_annotations.coco.json", root_dir + "test")
    
    # Define model configuration
    # Reference CFG parameters: https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda:3" # Set GPU or CPU devices
    cfg.OUTPUT_DIR = output_directory
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("facade_train_city",)
    cfg.DATASETS.TEST = ("facade_valid_city",)
    # cfg.TEST.EVAL_PERIOD = 100 # Done via Hooks (Iterations until validation is performed)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = [] # Learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 + 1  # Number of classes + background
    
    # Create the output directory if not exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

def log_predictions(predictor, dataset_dicts, experiment):
    """Log Model Predictions to Comet for analysis.

    Args:
        predictor (DefaultPredictor): Predictor Object for Detectron Model
        dataset_dicts (dict): Dataset Dictionary contaning samples of data and annotations
        experiment (comet_ml.Experiment): Comet Experiment Object
    """
    predictions_data = {}
    for d in random.sample(dataset_dicts, 3):
        file_name = str(d["file_name"])

        im = cv2.imread(file_name)
        annotations = d["annotations"]

        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        formatted_data = format_predictions(outputs, annotations)
        predictions_data[file_name] = formatted_data
        experiment.log_image(file_name, name=file_name)

    experiment.log_asset_data(predictions_data, name="predictions-data.json")


def main():
    
    clear_gpu_cache()
    
    experiment = comet_ml.Experiment()
    cfg = setup('./output/')

    # Wrap the Detectron Default Trainer
    trainer = CometDefaultTrainer(cfg, experiment)
    trainer.resume_or_load(resume=False)

    # Register Hook to compute metrics using an Evaluator Object
    trainer.register_hooks(
        [hooks.EvalHook(100, lambda: trainer.evaluate_metrics(cfg, trainer.model))]
    )

    # Register Hook to compute eval loss
    trainer.register_hooks(
        [hooks.EvalHook(100, lambda: trainer.evaluate_loss(cfg, trainer.model))]
    )
    trainer.train()

    # Evaluate Model Predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Custom testing threshold
    predictor = DefaultPredictor(cfg)
    log_predictions(predictor, get_facade_dicts("/val"), experiment)

if __name__ == "__main__":
    
    main()