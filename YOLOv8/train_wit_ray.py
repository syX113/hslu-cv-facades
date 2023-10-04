from ultralytics import YOLO
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

model = YOLO("yolov8n-seg.pt")
data = './YOLOv8/building-facade-segmentation-instance-1/config.yaml'

def model_train(config):
    trained_model = model.train(data=data, 
                            batch=16, 
                            epochs=2, 
                            workers=8, 
                            imgsz=1024, 
                            patience=1, 
                            save=False, 
                            device=[0,1,2,3],
                            project='YOLOv8',
                            name=f'./building-facade-segmentation-instance-1/runs/YOLOv8_ray/train'
                            )
                            
    return {'acc': 0.8}


ray.init()

# Define the search space for hyperparameters
search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8, 16, 32])
}

# Define the hyperparameter search algorithm
search_alg = HyperOptSearch(metric="acc", mode="max")

# Define the scheduler
scheduler = ASHAScheduler(max_t=10, grace_period=1)

# Perform hyperparameter tuning
analysis = tune.run(
    model_train,
    config=search_space,
    num_samples=10,  # Number of hyperparameter combinations to try
    search_alg=search_alg,
    scheduler=scheduler
)

# Get the best hyperparameters and evaluation metric
best_hyperparameters = analysis.get_best_config(metric="acc", mode="max")
best_metric = analysis.best_result["acc"]

print("Best hyperparameters:", best_hyperparameters)
print("Best evaluation metric:", best_metric)

# Clean up Ray resources
ray.shutdown()