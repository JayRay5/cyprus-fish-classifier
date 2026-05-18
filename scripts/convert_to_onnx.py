import os
import hydra
from omegaconf import DictConfig
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Safely extract configurations from Hydra
    model_id = cfg.model.target_hf_repo_id

    # Fallback to "model_onnx" if onnx_dir is not specified in your yaml configuration
    onnx_folder_name = cfg.model.get("onnx_dir", "model_onnx")

    # 2. CRITICAL FIX: Hydra automatically changes the current working directory (CWD)
    # to a timed output folder (e.g., outputs/2026-05-18/...).
    # We use get_original_cwd() to force the ONNX model to be saved at your true project root.
    project_root = hydra.utils.get_original_cwd()
    save_dir = os.path.join(project_root, onnx_folder_name)

    print(f"⏳ Downloading and converting '{model_id}' to ONNX...")
    print(f"📁 Target local directory: {save_dir}")

    # 3. Load PyTorch model and export it to ONNX automatically (export=True)
    model = ORTModelForImageClassification.from_pretrained(model_id, export=True)  # nosec B615

    # 4. Load the image processor (for normalization, resizing, etc.)
    processor = AutoImageProcessor.from_pretrained(model_id)  # nosec B615

    # 5. Save the ONNX model and processor configuration to the correct root folder
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"✅ Model successfully converted and saved to: '{save_dir}'")


if __name__ == "__main__":
    main()
