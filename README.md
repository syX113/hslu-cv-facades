# Computer Vision (Façade Segmentation Optimization)

* [Introduction](#introduction)
* [Research Questions](#research-questions)
* [Underlying Data](#underlying-data)
* [Proceeding & Methodologies](#proceeding---methodologies)
* [Results](#results)
* [Authors](#authors)

## Background 

In 2021, Switzerland launched a comprehensive climate strategy aimed at achieving net-zero emissions by 2050, with specific reduction targets in sectors such as buildings, transport, and agriculture.\
The building sector is a focal point as it consumes about 40% of the nation's total energy and emits one-third of its domestic CO2. To meet the 2050 targets, the country needs to gather improved data on the condition of buildings and their owners,focusing especially on thermal insulation quality.

Traditional assessment methods like thermal imaging face challenges due to the large number of buildings and limited equipment.
However, infrared thermography has emerged as a promising technique for large-scale assessments, helping to identify buildings with poor insulation and high energy loss.

To conduct these large-scale assessments effectively, a critical preliminary step involves segmenting the building façade into different parts, e.g. window, balcony and other parts. 
This segmentation allows for a more targeted approach when using thermal image analysis, focusing only on the effective areas of the building façade.

![image](https://github.com/syX113/hslu-cv-facades/assets/118247293/8430ac29-b550-464f-bb34-3ec3128c3f91)


## Introduction 

In initial, heuristic tests, two deep learning methods showed promising results for segmenting building facades:\
1. Mask R-CNN ([Detectron2 implementation from Meta/FAIR](https://ai.meta.com/tools/detectron2/))
2. YOLOv8 ([from Ultralytics](https://docs.ultralytics.com))

The primary objective of this research project is to identify the most effective deep learning method to segment the buildings into: \
Façade \
Window \
Balcony \
...

This will involve optimizing both approaches through additional image preparation steps and then comparing and validating these methods using appropriate metrics and techniques.
![image](https://github.com/syX113/hslu-cv-facades/assets/118247293/f19e31ad-5728-46f1-b7cf-ccf46be06ae2)

## Research Questions 

The research questions are as follows:

1. *Do CLAHE, greyscale, and built-in augmentation techniques influence the mAP50, segmentation, and class loss of deep learning models Detectron2 (Mask R-CNN) and Ultralytics (YOLOv8) in the task of building façade segmentation?*
2. *To what extent do optimized hyperparameters enhance the performance of Detectron2 (Mask R-CNN) and Ultralytics (YOLOv8) in segmenting building façades in terms of mAP50, segmentation, and class loss?*
3. *In the context of building façade segmentation, what is the comparative effectiveness of Detectron2 (Mask R-CNN) and Ultralytics (YOLOv8) in terms of mAP50, segmentation and class loss?*

The following metrics were used to evaluate model performance: 
- Loss: A measure of the model's optimization performance, with lower values indicating better convergence during training.
- Segmentation Loss: Similar to the loss, focusing on how well the model segments objects in the validation dataset.
- mAP50 (segmentation mask): mAP (mean Average Precision) at 50% Intersection over Union (IoU) threshold. Measures precision and recall for segmentation tasks with higher values indicating better segmentation performance..

These metrics are selected based on:
- Quantitative Assessment: mAP50 and loss metrics offer a numerical way to evaluate the performance of segmentation models. mAP50 assesses object segmentation, while loss measures optimization progress.
- Objective Comparison: Loss metrics provide an objective, consistent basis for comparing segmentation models, aiding in selecting the best one.
- Informative Evaluation: Multiple metrics offer a comprehensive view of segmentation model performance, helping understand trade-offs and make informed decisions.


## Underlying Data

The dataset is hosted on [Roboflow - Project: building-facade-segmentation-instance](https://universe.roboflow.com/building-facade/building-facade-segmentation-instance).\
It contains 598 annotated images of building façades.

Classes:
- balcony-fence
- car
- facade
- fence
- non-building-infrastructure
- shop
- street
- traffic-infrastructure
- vegetation
- window

Additional datasets are not considered and used for testing/validation.

## Detectron2 in-depth

Part 1 to 5:
- https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd
- https://medium.com/@hirotoschwert/digging-into-detectron-2-part-2-dd6e8b0526e
- https://medium.com/@hirotoschwert/digging-into-detectron-2-part-3-1ecc27efc0b2
- https://medium.com/@hirotoschwert/digging-into-detectron-2-part-4-3d1436f91266
- https://medium.com/@hirotoschwert/digging-into-detectron-2-part-5-6e220d762f9


## Methodology & Proceeding 

Visual and exploratory dataset analysis was initially performed through a Jupyter Notebook. \
Subsequently, the following steps were executed for both YOLO and Mask-R-CNN frameworks:

- Training the "base" model.
- Training with CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancements.
- Training in greyscale.
- Selection of the best "base" model based on AP50 (Average Precision at 50% Intersection over Union).
-Incorporating framework augmentations, such as flip, crop, and rotation.
- Comparing the performance of the best base model with the framework-augmented model, again using AP50 as the metric.
- Applying hyperparameter tuning to further optimize the best model's AP50.
- Training a YOLO model with the best-performing hyperparameters.
- Comparing the performance of YOLOv8 with the Mask-R-CNN model, once again using AP50 as the evaluation metric.

All of these steps were executed for both YOLO and Mask-R-CNN frameworks. A train/valid/test split was applied consistently during these training phases, with k-fold cross-validation planned for later stages which has not been implemented so far. 

![image](https://github.com/syX113/hslu-cv-facades/assets/118247293/e05f4602-5718-47ea-8f70-c28979976680)

## Results

Results were visualized in CometML and can be found under: https://www.comet.com/syx/hslu-computer-vision/reports/hslu-computer-vision-facade-segmentation \

The comparison between YOLOv8 models (Base, CLAHE (Contrast Limited Adaptive Histogram Equalization), and Grayscale versions) revealed the following findings:
- mAP50: The mAP50 scores were relatively equal among the models, with the base model slightly outperforming the others.
- Loss: The loss values for all three models were equal.
- Seg_loss: Segmentation loss was also equal across all models, but each exhibited strong overfitting tendencies. To address this issue, it was determined that augmentation methods should be applied.
![image](https://github.com/syX113/hslu-cv-facades/assets/118247293/0ab18a68-ef9f-4745-b2e9-352b731395ab)

The comparison between the YOLOv8 Base model and the Augmented model yielded the following insights:
- mAP50: The augmented model outperformed the base model, achieving a higher mAP50 score.
- Loss: The base model exhibited a faster decay in its loss function during training.
- Seg_loss: The base model showed signs of strong overfitting, while the augmented model displayed a faster decay in segmentation loss.
![image](https://github.com/syX113/hslu-cv-facades/assets/118247293/8d66b453-0028-448e-a154-5460f6620fcd)

As a result, it was observed that the augmented model required more training time but offered superior performance in the segmentation task and demonstrated reduced overfitting to the dataset. \



Overall, the YOLOv8 Augmented Model performed better than the Mask R-CNN Model in the observed metrics. 

## Authors

Lukas Zurbriggen & Tim Giger\
*Hochschule Luzern, M.Sc. in Applied Information & Data Science*\
*Module: Computer Vision*
