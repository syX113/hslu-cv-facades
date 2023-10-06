#!/usr/bin/env python

import os
import comet_ml
# Import custom trainer & utils
from detectron2.engine import DefaultPredictor, hooks
from trainer import CustomDetectronTrainer
from model_config import get_mask_config
from utils.training_utils import init_gpu, log_predictions

def train_mask():
       
    # Create the configuration and Comet experiment
    cfg = get_mask_config(default_augs=False, greyscale=False, clahe=False, sampling=False)
    
    experiment = create_experiment()

    # Wrap the Detectron Default Trainer and initialize trainer instance with defined hooks
    trainer = CustomDetectronTrainer(cfg, experiment)
    setup_hooks(trainer, cfg)
    
    # Start the training and evaluate the model
    trainer.train()
    evaluate_model(cfg, experiment)

def create_experiment():
    
    # Read Comet API key
    with open('mask-r-cnn/utils/comet_api_key.txt', 'r') as file:
        api_key = file.read()
    os.environ['COMET_API_KEY'] = api_key
    os.environ['COMET_PROJECT_NAME'] = 'hslu-computer-vision'
    
    # Initialize logging with Comet and start new experiment
    comet_ml.init()
    return comet_ml.Experiment()
    
def setup_hooks(trainer, cfg):
    trainer.resume_or_load(resume=False)
    # Register Hook to compute metrics / loss using an Evaluator Object
    trainer.register_hooks([hooks.EvalHook(100, lambda: trainer.evaluate_metrics(cfg, trainer.model))])
    trainer.register_hooks([hooks.EvalHook(100, lambda: trainer.evaluate_loss(cfg, trainer.model))])
    
def evaluate_model(cfg, experiment): 
    # Evaluate Model Predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Custom testing threshold
    predictor = DefaultPredictor(cfg)
    log_predictions(predictor, experiment, 'facade_test_city')
    

if __name__ == "__main__":
    
    # Clear GPU cache and get CUDA/Torch versions
    init_gpu()
    # Start the training loop
    train_mask()