import random
import cv2
import torch
import detectron2
from GPUtil import showUtilization as gpu_usage
from fvcore.common.config import CfgNode
from detectron2.data import DatasetCatalog


def init_gpu():

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    print("Initial GPU Usage: \n")
    gpu_usage()
    print("GPU Usage after emptying the cache: \n")
    torch.cuda.empty_cache()
    gpu_usage()
    
# Original Source: https://github.com/comet-ml/comet-detectron/blob/main/train.py
# Adjusted and refactored to meet the requirements
def log_predictions(predictor, experiment, dataset_name="facade_test_city"):
    """Log Model Predictions to Comet for analysis.

    Args:
        predictor (DefaultPredictor): Predictor Object for Detectron Model
        experiment (comet_ml.Experiment): Comet Experiment Object
        dataset_name (str): Name of the registered dataset to use. Default is "facade_test_city".
    """
    # Retrieve the dataset dictionary using DatasetCatalog
    dataset_dicts = DatasetCatalog.get(dataset_name)

    predictions_data = {}
    for d in random.sample(dataset_dicts, 3):
        file_name = str(d["file_name"])

        im = cv2.imread(file_name)
        annotations = d["annotations"]

        outputs = predictor(
            im
        )  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        formatted_data = format_predictions(outputs, annotations)
        predictions_data[file_name] = formatted_data
        experiment.log_image(file_name, name=file_name)

    experiment.log_asset_data(predictions_data)
    

# Original Source: https://github.com/comet-ml/comet-detectron/blob/main/utils.py
# Adjusted and refactored to meet the requirements
def format_predictions(outputs, annotations):
    """Format Dectectron Predictions so that they can be visualized using
    Comet Panels

    Args:
        outputs (list): List of formatted dicts
    """
    data = []
    prediction = outputs["instances"].to("cpu")

    predicted_boxes = prediction.pred_boxes.tensor.numpy().tolist()
    predicted_scores = prediction.scores.numpy().tolist()
    predicted_classes = prediction.pred_classes.numpy().tolist()

    for annotation in annotations:
        bbox = annotation["bbox"]

        # Convert from numpy.int64 to int, if needed
        x, y, x2, y2 = map(lambda x: x.item() if hasattr(x, 'item') else x, bbox)

        label = annotation["category_id"]
        data.append(
            {
                "label": f"ground_truth-{label}",
                "score": 100,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
            }
        )

    for predicted_box, predicted_score, predicted_class in zip(
        predicted_boxes, predicted_scores, predicted_classes
    ):
        x, y, x2, y2 = predicted_box
        data.append(
            {
                "label": predicted_class,
                "box": {"x": x, "y": y, "x2": x2, "y2": y2},
                "score": predicted_score * 100,
            }
        )

    return data

# Original Source: https://github.com/comet-ml/comet-detectron/blob/main/comet_trainer.py
# Adjusted and refactored to meet the requirements
def log_config(cfg, experiment):
    """Traverse the Detectron Config graph and log the parameters

    Args:
        cfg (CfgNode): Detectron Config Node
        experiment (comet_ml.Experiment): Comet ML Experiment object
    """

    def log_node(node, prefix):
        if not isinstance(node, CfgNode):
            if isinstance(node, dict):
                experiment.log_parameters(node, prefix=prefix)

            else:
                experiment.log_parameter(name=prefix, value=node)
            return

        node_dict = dict(node)
        for k, v in node_dict.items():
            _prefix = f"{prefix}-{k}" if prefix else k
            log_node(v, _prefix)

    log_node(cfg, "")