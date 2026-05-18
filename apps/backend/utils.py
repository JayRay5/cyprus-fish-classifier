import numpy as np


def predict_image(image, processor, model):
    """
    Predict fish species using the model (ONNX or PyTorch) and NumPy.
    """
    # 1. Prepare image inputs as NumPy arrays
    inputs = processor(images=image, return_tensors="np")

    # 2. Model inference
    outputs = model(**inputs)
    logits = outputs.logits[0]

    # 3. MLOps fallback: Convert PyTorch tensor to NumPy array if returned (e.g., during unit tests)
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    else:
        logits = np.asarray(logits)

    # 4. Compute stable softmax using NumPy
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    # 5. Format results using model id2label configuration
    results = {}
    for idx, prob in enumerate(probabilities):
        label_name = model.config.id2label[idx]
        results[label_name] = float(prob.item())

    # 6. Sort results by probability in descending order
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_results
