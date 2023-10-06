import comet_ml
from ultralytics import YOLO
from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import gc


# run program in back:
# nohup /home/ubuntu/miniconda3/envs/venv-yolov8/bin/python /home/ubuntu/hslu-computer-vision/hslu-cv-facades/YOLOv8/building-facade-segmentation-instance-1/train_yolov8.py &
# kill process that blocks cpu: 
#sudo kill -9 462580 <- process number (can be seen after nvidia-smi)
#or
# sudo fuser -k -9 /dev/nvidia0 <- replace 0 with GPU index 

# set up comet ml logging 
comet_ml.init(project_name='hslu-computer-vision')


def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()

#---------------------------------------------------------------------------------------------------------------------------------
    
def do_train_base():
    #first disable albumentations! pip uninstall -y albumentations
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml'
                ,epochs=10 
                ,imgsz=1024 
                ,batch=8 # AutoBatch?
                ,plots=True 
                ,device=[0,1] #change cores beeing used  
                ,save_period=100 
                ,workers=4
                ,val=True
                ,project='YOLOv8'
                ,close_mosaic = 0 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=False
                ,lr0= 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                ,lrf= 0.01  # (float) final learning rate (lr0 * lrf)
                ,momentum= 0.937  # (float) SGD momentum/Adam beta1
                ,weight_decay= 0.0005  # (float) optimizer weight decay 5e-4
                ,warmup_epochs= 3.0  # (float) warmup epochs (fractions ok)
                ,warmup_momentum= 0.8  # (float) warmup initial momentum
                ,warmup_bias_lr= 0.1  # (float) warmup initial bias lr
                ,box= 7.5  # (float) box loss gain
                ,cls= 0.5  # (float) cls loss gain (scale with pixels)
                ,dfl= 1.5  # (float) dfl loss gain
                ,pose= 12.0  # (float) pose loss gain
                ,kobj= 1.0  # (float) keypoint obj loss gain
                ,label_smoothing= 0.0  # (float) label smoothing (fraction)
                ,hsv_h= 0 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.0  # (float) image rotation (+/- deg)
                ,translate= 0  # (float) image translation (+/- fraction)
                ,scale= 0  # (float) image scale (+/- gain)
                ,shear= 0  # (float) image shear (+/- deg)
                ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0  # (float) image flip up-down (probability)
                ,fliplr= 0  # (float) image flip left-right (probability)
                ,mosaic= 0  # (float) image mosaic (probability)
                ,mixup= 0.0  # (float) image mixup (probability)
                ,copy_paste= 0.0  # (float) segment copy-paste (probability)
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_base/train')
    
    
#---------------------------------------------------------------------------------------------------------------------------------


def do_train_grayscale():
    #first disable albumentations! pip uninstall -y albumentations
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config_grayscale.yaml'
                ,epochs=1000 
                ,imgsz=1024 
                ,batch=8 # AutoBatch?
                ,plots=True 
                ,device=[0,1,2,3] #change cores beeing used  
                ,save_period=100 
                ,workers=4
                ,val=True
                ,project='YOLOv8'
                ,close_mosaic = 0 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=False
                ,lr0= 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                ,lrf= 0.01  # (float) final learning rate (lr0 * lrf)
                ,momentum= 0.937  # (float) SGD momentum/Adam beta1
                ,weight_decay= 0.0005  # (float) optimizer weight decay 5e-4
                ,warmup_epochs= 3.0  # (float) warmup epochs (fractions ok)
                ,warmup_momentum= 0.8  # (float) warmup initial momentum
                ,warmup_bias_lr= 0.1  # (float) warmup initial bias lr
                ,box= 7.5  # (float) box loss gain
                ,cls= 0.5  # (float) cls loss gain (scale with pixels)
                ,dfl= 1.5  # (float) dfl loss gain
                ,pose= 12.0  # (float) pose loss gain
                ,kobj= 1.0  # (float) keypoint obj loss gain
                ,label_smoothing= 0.0  # (float) label smoothing (fraction)
                ,hsv_h= 0 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.0  # (float) image rotation (+/- deg)
                ,translate= 0  # (float) image translation (+/- fraction)
                ,scale= 0  # (float) image scale (+/- gain)
                ,shear= 0  # (float) image shear (+/- deg)
                ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0  # (float) image flip up-down (probability)
                ,fliplr= 0  # (float) image flip left-right (probability)
                ,mosaic= 0  # (float) image mosaic (probability)
                ,mixup= 0.0  # (float) image mixup (probability)
                ,copy_paste= 0.0  # (float) segment copy-paste (probability)
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_grayscale/train')


#---------------------------------------------------------------------------------------------------------------------------------



def do_train_CLAHE():
    #first disable albumentations! pip uninstall -y albumentations
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config_CLAHE.yaml'
                ,epochs=1000 
                ,imgsz=1024 
                ,batch=8 # AutoBatch?
                ,plots=True 
                ,device=[0,1] #change cores beeing used  
                ,save_period=100 
                ,workers=4
                ,val=True
                ,project='YOLOv8'
                ,close_mosaic = 0 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=False
                ,lr0= 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                ,lrf= 0.01  # (float) final learning rate (lr0 * lrf)
                ,momentum= 0.937  # (float) SGD momentum/Adam beta1
                ,weight_decay= 0.0005  # (float) optimizer weight decay 5e-4
                ,warmup_epochs= 3.0  # (float) warmup epochs (fractions ok)
                ,warmup_momentum= 0.8  # (float) warmup initial momentum
                ,warmup_bias_lr= 0.1  # (float) warmup initial bias lr
                ,box= 7.5  # (float) box loss gain
                ,cls= 0.5  # (float) cls loss gain (scale with pixels)
                ,dfl= 1.5  # (float) dfl loss gain
                ,pose= 12.0  # (float) pose loss gain
                ,kobj= 1.0  # (float) keypoint obj loss gain
                ,label_smoothing= 0.0  # (float) label smoothing (fraction)
                ,hsv_h= 0 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.0  # (float) image rotation (+/- deg)
                ,translate= 0  # (float) image translation (+/- fraction)
                ,scale= 0  # (float) image scale (+/- gain)
                ,shear= 0  # (float) image shear (+/- deg)
                ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0  # (float) image flip up-down (probability)
                ,fliplr= 0  # (float) image flip left-right (probability)
                ,mosaic= 0  # (float) image mosaic (probability)
                ,mixup= 0.0  # (float) image mixup (probability)
                ,copy_paste= 0.0  # (float) segment copy-paste (probability)
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_CLAHE/train')


#---------------------------------------------------------------------------------------------------------------------------------



def do_train_augmented():
    #first enable albumentations! pip install albumentations

    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml', 
                epochs=1000, 
                imgsz=1024, 
                batch=8, # AutoBatch?
                plots=True, 
                device=[0,1], #change cores beeing used  
                save_period=100, 
                workers=4,
                val=True,
                augment=True,
                project='YOLOv8',
                name='./building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train')
        
    
def do_train_yolov8_lr():
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml', 
                epochs=1000, 
                imgsz=1024, 
                batch=8, # AutoBatch?
                plots=True, 
                device=[0,1], #change cores beeing used  
                save_period=100, 
                workers=4,
                val=True,
                augment=True,
                lr0 = 0.1,
                lrf = 0.00001,
                project='YOLOv8',
                name='./building-facade-segmentation-instance-1/runs/YOLOv8_lr/train')

def do_validation():
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train/weights/best.pt')  # load model
    results = model.val(split='val',
                        project='YOLOv8',
                        name=f'./building-facade-segmentation-instance-1/runs/YOLOv8_augmented/val') # evaluate model performance on the validation set
                          
    results.box.map    # map50-95
    results.box.map50  # map50
    results.box.map75  # map75
    results.box.maps   # a list contains map50-95 of each category
    
    
def do_prediction():
    
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train/weights/best.pt')  # load model
    
    # Define the folder with pictures to apply predictions
    model.predict('/home/ubuntu/data/unzipped/facade-original-yolo-segmentation/test/images', 
                  save=True, 
                  imgsz=1024, 
                  conf=0.1, 
                  device=['CPU'], #use CPU because else CUDA out of memory for all the 83 predicitions
                  project='YOLOv8',
                  name='./building-facade-segmentation-instance-1/runs/YOLOv8_augmented/predict')
    

def do_tune(): 
    
    model = YOLO('yolov8n-seg.pt')  # load model    
    model.tune(data='config.yaml', epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)


        



if __name__ == '__main__':
    clean_mem()
    #do_train_base()
    #do_train_grayscale()
    do_train_CLAHE()
    