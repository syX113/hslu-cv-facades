import comet_ml
from ultralytics import YOLO
from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import gc

def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()

# run program in back:
# nohup /home/ubuntu/miniconda3/envs/venv-yolov8/bin/python /home/ubuntu/hslu-computer-vision/hslu-cv-facades/YOLOv8/train_yolov8.py &

# kill process that blocks cpu: 
#sudo kill -9 462580 <- process number (can be seen after nvidia-smi)
#or
# sudo fuser -k -9 /dev/nvidia0 <- replace 0 with GPU index 

# set up comet ml logging 
comet_ml.init(project_name='hslu-computer-vision')
#experiment = comet_ml.Experiment()
# Set the name of the experiment
#experiment.set_name(name='yolov8_base')



def do_train():
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt')

    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/data.yaml', 
                epochs=1000, 
                imgsz=1024, 
                batch=8, # AutoBatch?
                plots=True, 
                device=[0,1,2,3], #change cores beeing used  
                save_period=100, 
                workers=4,
                val=True,
                project='YOLOv8',
                name='./building-facade-segmentation-instance-1/runs/YOLOv8_base/train')

def do_validation():
    
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_base/train/weights/best.pt')  # load model
    results = model.val(split='val',
                        project='YOLOv8',
                        name='./building-facade-segmentation-instance-1/runs/YOLOv8_base/val') # evaluate model performance on the validation set
                          
    results.box.map    # map50-95
    results.box.map50  # map50
    results.box.map75  # map75
    results.box.maps   # a list contains map50-95 of each category
    
    
def do_prediction():
    
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_base/train/weights/best.pt')  # load model
    
    # Define the folder with pictures to apply predictions
    model.predict('/home/ubuntu/data/unzipped/facade-original-yolo-segmentation/test/images', 
                  save=True, 
                  imgsz=1024, 
                  conf=0.1, 
                  device=['CPU'], #use CPU because else CUDA out of memory for all the 83 predicitions
                  project='YOLOv8',
                  name='./building-facade-segmentation-instance-1/runs/YOLOv8_base/predict')
    

    


if __name__ == '__main__':
    do_prediction()
    