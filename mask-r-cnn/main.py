import os
# Setup & initialize Comet logging
with open('mask-r-cnn/comet_api_key.txt', 'r') as file:
    api_key = file.read()
os.environ['COMET_API_KEY'] = api_key
os.environ['COMET_PROJECT_NAME'] = 'hslu-computer-vision'

import comet_ml
from comet_trainer import CometDefaultTrainer
from train_mask_r_cnn import setup, log_predictions
from utils import clear_gpu_cache
from detectron2.engine import DefaultPredictor, hooks

def main():
    
    # Clear GPU cache to prevent CUDA memory overflow
    clear_gpu_cache()
    
    # Initialize logging with Comet
    comet_ml.init()
    experiment = comet_ml.Experiment()
    
    # Create the configuration
    cfg = setup('./mask-r-cnn/output/')

    # Wrap the Detectron Default Trainer and initialize trainer instance
    trainer = CometDefaultTrainer(cfg, experiment)
    trainer.resume_or_load(resume=False)

    # Register Hook to compute metrics / loss using an Evaluator Object
    trainer.register_hooks([hooks.EvalHook(100, lambda: trainer.evaluate_metrics(cfg, trainer.model))])
    trainer.register_hooks([hooks.EvalHook(100, lambda: trainer.evaluate_loss(cfg, trainer.model))])
    
    # Start the training
    trainer.train()
    
    # Evaluate Model Predictions
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Custom testing threshold
    predictor = DefaultPredictor(cfg)
    log_predictions(predictor, experiment, 'facade_test_city')

if __name__ == "__main__":
    
    main()
    
    '''
    
    # Used for multi GPU i.e. distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # Execute training loop, using Detectron's launch method to support multi GPU
    launch(
        main_func=main,
        num_gpus_per_machine=3,
        num_machines=1,
        machine_rank=0,
        dist_url=None,
        args=(),
        timeout=timedelta(minutes=30)
    )
    '''