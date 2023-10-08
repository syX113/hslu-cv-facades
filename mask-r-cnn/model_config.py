import os
import statistics
from detectron2 import model_zoo
from detectron2.config import get_cfg
from utils.model_utils import class_sampling
from detectron2.data.datasets import register_coco_instances
# Add custom nodes to the model config
from detectron2.config import CfgNode as CN

def get_mask_config(output_directory='./mask-r-cnn/_output/', default_augs=False, custom_augs=None, sampling=False, dataset_dir = '../../data/unzipped/facade-original-coco-segmentation/'):

    # Register train & validation in COCO format
    register_coco_instances("facade_train_city", {}, dataset_dir + "train/_annotations.coco.json", dataset_dir + "train")
    register_coco_instances("facade_valid_city", {}, dataset_dir + "valid/_annotations.coco.json", dataset_dir + "valid")
    # Test dataset used later for the model evaluation but not used during training run (Hold-out strategy)
    register_coco_instances("facade_test_city", {}, dataset_dir + "test/_annotations.coco.json", dataset_dir + "test")
    
    # Define model configuration
    # Reference CFG parameters: https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda:2" # Set GPU or CPU devices, e.g. "cuda:3" to use specific GPU
    cfg.OUTPUT_DIR = output_directory
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("facade_train_city",)
    cfg.DATASETS.TEST = ("facade_valid_city",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = [] # Learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10 + 1  # Number of classes + background
    
    # Turn off default augmentations
    if default_augs:
        cfg.INPUT.RANDOM_FLIP = 'none'
        cfg.INPUT.CROP.ENABLED = False
        
    if sampling:
        repeat_factors = class_sampling('facade_train_city')
        # Sampling based on the computed class distributions
        cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        cfg.DATALOADER.REPEAT_THRESHOLD = statistics.fmean(repeat_factors)
        
    # Initialize AUGMENTATION.CUSTOM_LIST to add the custom augmentations to the config
    cfg.AUGMENTATION = CN()
    
    if custom_augs is None:
        custom_augs = []
        
    cfg.AUGMENTATION.CUSTOM_LIST = custom_augs
    
    # Create the output directory if not exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg