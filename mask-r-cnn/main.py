#!/usr/bin/env python

import os
import comet_ml
from detectron2.engine import DefaultPredictor, hooks
from trainer import CustomDetectronTrainer
from model_config import get_mask_config, register_datasets
from utils.training_utils import init_gpu, log_predictions
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import optuna

# Define the Objective function for Optuna
def objective(cfg, trial):
    
    # Define the hyperparameter search space
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    step1 = trial.suggest_int("step1", 500, 5000, step=500)
    step2 = trial.suggest_int("step2", step1+500, 10000, step=500)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.99)
    batch_size_per_image = trial.suggest_categorical("batch_size_per_image", [64, 128, 256, 512])
    anchor_sizes = trial.suggest_categorical("anchor_sizes", [[32, 64, 128], [64, 128, 256], [128, 256, 512]])
    warmup_factor = trial.suggest_float("warmup_factor", 0.01, 1)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 1000, step=100)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    
    # Overwriting default model config
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.STEPS = [step1, step2]
    #cfg.SOLVER.MAX_ITER = 1500 # Reducing to 1500, so the training is faster
    cfg.SOLVER.STEPS = (1000,2000) # Reduces LR at step 500 and 1000
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [anchor_sizes]
    cfg.SOLVER.WARMUP_FACTOR = warmup_factor
    cfg.SOLVER.WARMUP_ITERS = warmup_iters

    experiment = create_experiment()
    experiment.log_parameters(trial.params)

    trainer = CustomDetectronTrainer(cfg, experiment)
    setup_hooks(trainer, cfg)

    trainer.train()
    
    # Using AP metric as objective for optimization. 
    segm_ap = evaluate_model(cfg, experiment)
    experiment.log_metric("validation_segm_ap", segm_ap)
    
    return segm_ap


def hp_tuning_train_mask(cfg):
    
    # Clear GPU cache and get CUDA/Torch versions
    init_gpu()
    
    # Wrapper for the objective, so CFG is correctly passed
    def wrapped_objective(trial):
        return objective(cfg, trial)
    
    study = optuna.create_study(direction="maximize") # Using "maximize", since AP metric should be maximized  
    study.optimize(wrapped_objective, n_trials=25)

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

def train_mask(cfg):
    
    # Clear GPU cache and get CUDA/Torch versions
    init_gpu()
    # Create Comet Experiment to log all metrics
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
    
    # Evaluate on the validation set
    eval_output_dir = os.path.join(cfg.OUTPUT_DIR, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True) 
    evaluator = COCOEvaluator("facade_valid_city", cfg, False, output_dir=eval_output_dir)
    val_loader = build_detection_test_loader(cfg, "facade_valid_city")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    return results['segm']['AP']

if __name__ == "__main__":
    
    # Register the datasets for all runs
    register_datasets()
    
    # Train base model, without any augmentations
    train_mask(cfg = get_mask_config(default_augs=False, custom_augs=None, sampling=False))
    
    # Train model, only with CLAHE modifed images
    train_mask(cfg = get_mask_config(default_augs=False, custom_augs=["CustomAugmentationCLAHE"], sampling=False))
    
    # Train model, only with greyscale images
    train_mask(cfg = get_mask_config(default_augs=False, custom_augs=["CustomAugmentationGreyscale"], sampling=False))
    
    # Train model with Detectron's built-in augmentations (Flip/Crop)
    train_mask(cfg = get_mask_config(default_augs=True, custom_augs=None, sampling=False))
    
    # Start a hyperparameter tuning with 25 trials to find best parameters with built-in augmentations
    hp_tuning_train_mask(cfg = get_mask_config(default_augs=True, custom_augs=None, sampling=False))