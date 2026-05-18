import numpy as np


def predict_image(image, processor, model):
    """
    Prédit l'espèce du poisson en utilisant le modèle ONNX et Numpy.
    """
    # 1. Préparer l'image et demander des tableaux Numpy ("np") au lieu de PyTorch ("pt")
    inputs = processor(images=image, return_tensors="np")

    # 2. Inférence ONNX
    outputs = model(**inputs)
    logits = outputs.logits[0]  # Récupérer le premier (et unique) résultat du batch

    # 3. Calculer le Softmax manuellement avec Numpy
    # L'astuce "logits - np.max(logits)" évite les erreurs d'overflow
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    # 4. Formater les résultats avec id2label
    results = {}
    for idx, prob in enumerate(probabilities):
        label_name = model.config.id2label[idx]
        # On convertit le float32 de numpy en float natif Python avec .item()
        results[label_name] = float(prob.item())

    # Optionnel : Trier pour renvoyer les plus probables en premier
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_results
