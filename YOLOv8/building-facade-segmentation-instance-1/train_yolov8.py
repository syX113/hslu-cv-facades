# import libararies

import comet_ml
from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import gc
from pathlib import Path
from ray import tune



# set up comet ml logging 
comet_ml.init(project_name='hslu-computer-vision')

#function to first clean cuda memory
def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()

#---------------------------------------------------------------------------------------------------------------------------------
    
def do_train_base():
    #first disable albumentations! pip uninstall -y albumentations so that no basic augmentations are applied 
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml'
                ,epochs=3000 
                ,imgsz=1024 
                ,batch=8
                ,plots=True 
                ,device=[0,1,2,3] #change cores beeing used  
                ,save_period=250
                ,workers=4
                ,val=True
                ,project='YOLOv8'
                ,close_mosaic = 10 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=True
                #,lr0= 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                #,lrf= 0.01  # (float) final learning rate (lr0 * lrf)
                #,momentum= 0.937  # (float) SGD momentum/Adam beta1
                #,weight_decay= 0.0005  # (float) optimizer weight decay 5e-4
                ,warmup_epochs= 10  # (float) warmup epochs (fractions ok)
                #,warmup_momentum= 0.8  # (float) warmup initial momentum
                #,warmup_bias_lr= 0.1  # (float) warmup initial bias lr
                #,box= 7.5  # (float) box loss gain
                #,cls= 0.5  # (float) cls loss gain (scale with pixels)
                #,dfl= 1.5  # (float) dfl loss gain
                #,pose= 12.0  # (float) pose loss gain
                #,kobj= 1.0  # (float) keypoint obj loss gain
                #,label_smoothing= 0.0  # (float) label smoothing (fraction)
                ,hsv_h= 0.1 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0.1  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0.1  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.0  # (float) image rotation (+/- deg)
                ,translate= 0  # (float) image translation (+/- fraction)
                #,scale= 0  # (float) image scale (+/- gain)
                ,shear= 0  # (float) image shear (+/- deg)
                ,perspective= 0.001  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0.2  # (float) image flip up-down (probability)
                ,fliplr= 0.2  # (float) image flip left-right (probability)
                ,mosaic= 0.4  # (float) image mosaic (probability)
                ,mixup= 0.3  # (float) image mixup (probability)
                ,copy_paste= 0.2  # (float) segment copy-paste (probability)
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_base/train')
    
    
#---------------------------------------------------------------------------------------------------------------------------------


def do_train_grayscale():
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

def do_train_CLAHEbw2():
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config_CLAHEbw2.yaml'
                ,epochs=1000 
                ,imgsz=1024 
                ,batch=8 # AutoBatch?
                ,plots=True 
                ,device=[2,3] #change cores beeing used  
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
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_CLAHEbw2/train')


#---------------------------------------------------------------------------------------------------------------------------------

def do_train_CLAHEbw40():
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config_CLAHEbw40.yaml'
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
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_CLAHEbw40/train')


#---------------------------------------------------------------------------------------------------------------------------------

def do_train_CLAHEcol40():
    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('yolov8x-seg.pt') #yolov8x-seg
    # Train the model (adjust data.yaml with paths)
    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config_CLAHEcol40.yaml'
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
                ,name='./building-facade-segmentation-instance-1/runs/YOLOv8_CLAHEcol40/train')


#---------------------------------------------------------------------------------------------------------------------------------

def do_train_augmented():
    #first enable albumentations! pip install albumentations

    torch.cuda.empty_cache() 
    # Create new YOLO model
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_base/train/weights/best.pt') #best weights so far

    model.train(data='./YOLOv8/building-facade-segmentation-instance-1/config.yaml', 
                epochs=1000, 
                imgsz=1024, 
                batch=4,
                plots=True, 
                device=[1,2],
                save_period=100, 
                workers=4,
                val=True,
                project='YOLOv8',
                name='./building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train'
                ,close_mosaic = 10 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=True
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
                ,hsv_h= 0.015 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0.7  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0.4  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.1  # (float) image rotation (+/- deg)
                ,translate= 0.1  # (float) image translation (+/- fraction)
                ,scale= 0.5  # (float) image scale (+/- gain)
                ,shear= 0.25  # (float) image shear (+/- deg)
                ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0.5  # (float) image flip up-down (probability)
                ,fliplr= 0.5  # (float) image flip left-right (probability)
                ,mosaic= 0.5  # (float) image mosaic (probability)
                ,mixup= 0.25  # (float) image mixup (probability)
                ,copy_paste= 0.0)  # (float) segment copy-paste (probability)
        
#----------------------------------------------------------------

def hyp_tune(): 
    # Load a YOLOv8n model
    model = YOLO('yolov8n-seg.pt')
    result_grid = model.tune(data='/home/ubuntu/hslu-computer-vision/hslu-cv-facades/YOLOv8/building-facade-segmentation-instance-1/config.yaml',
               epochs=1000,
               space={"lr0": tune.uniform(1e-5, 1e-1),
                    "weight_decay": tune.uniform(1e-5, 1e-1),# (float) SGD momentum/Adam beta1
                    "momentum": tune.uniform(0.855, 0.999),
                    "batch": tune.choice([2, 4, 8])
                      }, 
                imgsz=1024, 
                plots=True, 
                save_period=100, 
                workers=4,
                project='YOLOv8',
                name='./building-facade-segmentation-instance-1/runs/YOLOv8_hyp/tune'
                ,close_mosaic = 10 #(int) disable mosaic augmentation for final epochs (0 to disable)
                ,augment=True
                ,lrf= 0.01  # (float) final learning rate (lr0 * lrf)
                ,warmup_epochs= 3.0  # (float) warmup epochs (fractions ok)
                ,warmup_momentum= 0.8  # (float) warmup initial momentum
                ,warmup_bias_lr= 0.1  # (float) warmup initial bias lr
                ,box= 7.5  # (float) box loss gain
                ,cls= 0.5  # (float) cls loss gain (scale with pixels)
                ,dfl= 1.5  # (float) dfl loss gain
                ,pose= 12.0  # (float) pose loss gain
                ,kobj= 1.0  # (float) keypoint obj loss gain
                ,label_smoothing= 0.0  # (float) label smoothing (fraction)
                ,hsv_h= 0.015 # (float) image HSV-Hue augmentation (fraction)
                ,hsv_s= 0.7  # (float) image HSV-Saturation augmentation (fraction)
                ,hsv_v= 0.4  # (float) image HSV-Value augmentation (fraction)
                ,degrees= 0.1  # (float) image rotation (+/- deg)
                ,translate= 0.1  # (float) image translation (+/- fraction)
                ,scale= 0.5  # (float) image scale (+/- gain)
                ,shear= 0.25  # (float) image shear (+/- deg)
                ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                ,flipud= 0.5  # (float) image flip up-down (probability)
                ,fliplr= 0.5  # (float) image flip left-right (probability)
                ,mosaic= 0.5  # (float) image mosaic (probability)
                ,mixup= 0.25  # (float) image mixup (probability)
                ,copy_paste= 0.0)  # (float) segment copy-paste (probability))



#----------------------------------------------------------------

def do_k_fold():
    model = YOLO('./YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_augmented/train/weights/best.pt') #best weights so far
    ksplit = 5
    results = {}

    # Create the list ds_yamls
    ds_yamls = [
        Path('./YOLOv8/building-facade-segmentation-instance-1/dataset_complete/2023-10-11_5-Fold_Cross-val/split_1/split_1_dataset.yaml'),
        Path('./YOLOv8/building-facade-segmentation-instance-1/dataset_complete/2023-10-11_5-Fold_Cross-val/split_2/split_2_dataset.yaml'),
        Path('./YOLOv8/building-facade-segmentation-instance-1/dataset_complete/2023-10-11_5-Fold_Cross-val/split_3/split_3_dataset.yaml'),
        Path('./YOLOv8/building-facade-segmentation-instance-1/dataset_complete/2023-10-11_5-Fold_Cross-val/split_4/split_4_dataset.yaml'),
        Path('./YOLOv8/building-facade-segmentation-instance-1/dataset_complete/2023-10-11_5-Fold_Cross-val/split_5/split_5_dataset.yaml')
    ]
    for k in range(ksplit):
        dataset_yaml = ds_yamls[k]
        model.train(data=dataset_yaml,                
                    epochs=10, 
                    imgsz=1024, 
                    batch=2, # AutoBatch?
                    plots=True, 
                    device=[1], #change cores beeing used  
                    save_period=100, 
                    workers=4,
                    val=True,
                    project='YOLOv8',
                    name='./building-facade-segmentation-instance-1/runs/YOLOv8_k_fold/kfold'
                    ,close_mosaic = 10 #(int) disable mosaic augmentation for final epochs (0 to disable)
                    ,augment=True
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
                    ,hsv_h= 0.015 # (float) image HSV-Hue augmentation (fraction)
                    ,hsv_s= 0.7  # (float) image HSV-Saturation augmentation (fraction)
                    ,hsv_v= 0.4  # (float) image HSV-Value augmentation (fraction)
                    ,degrees= 0.1  # (float) image rotation (+/- deg)
                    ,translate= 0.1  # (float) image translation (+/- fraction)
                    ,scale= 0.5  # (float) image scale (+/- gain)
                    ,shear= 0.25  # (float) image shear (+/- deg)
                    ,perspective= 0  # (float) image perspective (+/- fraction), range 0-0.001
                    ,flipud= 0.5  # (float) image flip up-down (probability)
                    ,fliplr= 0.5  # (float) image flip left-right (probability)
                    ,mosaic= 0.5  # (float) image mosaic (probability)
                    ,mixup= 0.25  # (float) image mixup (probability)
                    ,copy_paste= 0.0)  # (float) segment copy-paste (probability)
        results[k] = model.metrics  # save output metrics for further analysis



def do_validation():
    model = YOLO('/home/ubuntu/code/hslu-cv-facades/YOLOv8/building-facade-segmentation-instance-1/runs/YOLOv8_base/train7/weights/best.pt')  # load model
    results = model.val(split='test',
                        project='YOLOv8',
                        name=f'./building-facade-segmentation-instance-1/runs/YOLOv8_augmented/val',
                        plots=True) # evaluate model performance on the validation set
                          
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
    




#choose what do you want to do 
if __name__ == '__main__':
    clean_mem()
    do_validation()
    #do_train_grayscale()
    #do_train_CLAHEbw2()
    #do_train_CLAHEbw40()
    #do_train_CLAHEcol40()
    #do_train_augmented()
    #hyp_tune()
    #do_k_fold()
    #do_prediction()
