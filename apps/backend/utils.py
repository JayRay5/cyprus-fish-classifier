import torch

from transformers import PreTrainedModel, AutoImageProcessor
from PIL import Image


def predict_image(
    image: Image.Image, processor: AutoImageProcessor, model: PreTrainedModel
):
    """
    Take the image, processor and model -> return the probabilities of the input image for each classes
    """
    inputs = (
        processor(images=image, return_tensors="pt").to(model.device).to(model.dtype)
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    id2label = model.config.id2label
    results = {}

    for idx, prob in enumerate(probs):
        idx_int = idx
        label_name = id2label[idx_int]

        results[label_name] = float(prob)

    return results
