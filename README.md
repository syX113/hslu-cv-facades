# HSLU: Computer Vision (Façade Segmentation Optimization)

Table of contents:

* [Introduction](#introduction)
* [Research Questions](#research-questions)
* [Underlying Data](#underlying-data)
* [Proceeding & Methodologies](#proceeding---methodologies)
* [Results](#results)

## Background 

In 2021, Switzerland launched a comprehensive climate strategy aimed at achieving net-zero emissions by 2050, with specific reduction targets in sectors such as buildings, transport, and agriculture.\
The building sector is a focal point as it consumes about 40% of the nation's total energy and emits one-third of its domestic CO2. To meet the 2050 targets, the country needs to gather improved data on the condition of buildings and their owners,focusing especially on thermal insulation quality.

Traditional assessment methods like thermal imaging face challenges due to the large number of buildings and limited equipment.
However, infrared thermography has emerged as a promising technique for large-scale assessments, helping to identify buildings with poor insulation and high energy loss.

To conduct these large-scale assessments effectively, a critical preliminary step involves segmenting the building façade into different parts, e.g. window, balcony and other parts. 
This segmentation allows for a more targeted approach when using thermal image analysis, focusing only on the effective areas of the building façade.

## Introduction 

In initial, heuristic tests, two deep learning methods showed promising results for segmenting building facades:\
1. Mask R-CNN ([Detectron2 implementation from Meta/FAIR](https://ai.meta.com/tools/detectron2/))
2. YOLOv8 ([from Ultralytics](https://docs.ultralytics.com))

The primary objective of this research project is to identify the most effective deep learning method.\
This will involve optimizing both approaches through additional image preparation steps and then comparing and validating these methods using appropriate metrics and techniques.

## Research Questions 

The research questions are as follows:

1. *Which data preperation/augmentation steps have a positive effect on the Mean IoU for the building façade segmentation task on the specified dataset?*
2. *Which approach leads to a higher Mean IoU in the building façade segmentation task on the specified dataset?*

## Underlying Data

The dataset is hosted on [Roboflow - Project: building-facade-segmentation-instance](https://universe.roboflow.com/building-facade/building-facade-segmentation-instance).\ 
It contains 598 annotated images of building façades.\

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

## Methodologies & Proceeding

to do\

## Results 

to do\

### Authors

Lukas Zurbriggen & Tim Giger\
*Hochschule Luzern, M.Sc. in Applied Information & Data Science*\
*Module: Computer Vision*