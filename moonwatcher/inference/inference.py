import torch
from tqdm import tqdm
from moonwatcher.model.model import MoonwatcherModel
from moonwatcher.dataset.dataset import MoonwatcherDataset
from moonwatcher.utils.helpers import convert_to_list
from moonwatcher.utils.api_connector import upload_if_possible
from moonwatcher.utils.data import DataType, Task
from moonwatcher.annotations import (
    Predictions,
    PredictedLabels,
    PredictedBoundingBoxes,
    Labels,
    BoundingBoxes,
)


def inference(
    model: MoonwatcherModel, dataset: MoonwatcherDataset, device=torch.device("cpu")
):
    model.to(device=device)
    model.eval()

    predictions = Predictions(model=model, dataset=dataset)

    with tqdm(
        range(len(dataset)),
        desc=f"Running model {model.name} on dataset {dataset.name}",
        unit="Batches",
    ) as pbar:
        for id in pbar:
            task = model.task
            x = dataset[id][0]
            try:
                inputs = model.output_input_transform.transform_input(x)
            except Exception as e:
                raise Exception(f"Application of transform_input failed: {e}")

            try:
                if len(inputs) == 2:
                    args, kwargs = inputs
                    with torch.no_grad():
                        output = model(*args, **kwargs)
                else:
                    with torch.no_grad():
                        output = model(inputs)
            except Exception as e:
                raise Exception(
                    f"Input format after application of transform_input is invalid: {e}"
                )

            try:
                transformed_output = model.output_input_transform.transform_output(
                    output
                )
            except Exception as e:
                raise Exception(f"Application of transform_output failed: {e}")

            if task == Task.CLASSIFICATION.value:
                if len(transformed_output) == 1:
                    label_pred = transformed_output
                    prediction = Labels(datapoint_number=id, labels=label_pred)
                elif len(transformed_output) == 2:
                    label_pred, scores = transformed_output
                    prediction = PredictedLabels(
                        datapoint_number=id, labels=label_pred, scores=scores
                    )
                else:
                    raise ValueError(
                        "Length of transformed_output for classification should be 1 (labels) or 2 (labels and scores)!"
                    )
            elif task == Task.DETECTION.value:
                if len(transformed_output) == 2:
                    bounding_boxes_pred, labels_pred = transformed_output
                    prediction = BoundingBoxes(
                        datapoint_id=id,
                        boxes_xyxy=bounding_boxes_pred,
                        labels=labels_pred,
                    )
                elif len(transformed_output) == 3:
                    bounding_boxes_pred, labels_pred, scores = transformed_output
                    prediction = PredictedBoundingBoxes(
                        datapoint_number=id,
                        boxes_xyxy=bounding_boxes_pred,
                        labels=labels_pred,
                        scores=scores,
                    )
                else:
                    raise ValueError(
                        "Length of transformed_output for object detection must be 2 (bounding boxes and labels) or 3 "
                        "(bounding boxes, labels and scores)!"
                    )

            predictions.add(annotation=prediction)

    model.upload_if_not()

    predictions.store()
    _upload_predictions(
        predictions=predictions, dataset_name=dataset.name, model_name=model.name
    )


def _upload_predictions(predictions, dataset_name, model_name):
    data = []
    for prediction in predictions:
        data.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "datapoint_number": prediction.datapoint_number,
                "boxes": (
                    [convert_to_list(boxes) for boxes in prediction.boxes_xyxy]
                    if hasattr(prediction, "boxes_xyxy")
                    else None
                ),
                "labels": convert_to_list(prediction.labels),
                "scores": (
                    convert_to_list(prediction.scores)
                    if hasattr(prediction, "scores")
                    else None
                ),
            }
        )
    return upload_if_possible(datatype=DataType.PREDICTIONS.value, data=data)
