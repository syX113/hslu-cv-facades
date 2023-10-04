from ultralytics import YOLO
from ray import tune

# Load a YOLOv8n model
model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train/weights/best.pt')

# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml',
                         space={"lr0": tune.uniform(1e-5, 1e-1)},
                         epochs=50,
                         use_ray=True)