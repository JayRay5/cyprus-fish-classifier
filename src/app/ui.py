import gradio as gr
import os
import asyncio
from fastapi.concurrency import run_in_threadpool
from .utils import predict_image
from .config import settings


def create_ui(model_context: dict, model_lock: asyncio.Lock):
    async def predict_wrapper(image):
        if not image:
            return None

        if "model" not in model_context:
            raise gr.Error("Backend starting...")

        async with model_lock:
            results = await run_in_threadpool(
                predict_image, image, model_context["processor"], model_context["model"]
            )

        if max(results.values()) <= 0.3:
            return "Error: the fish has not been recognize."

        return results

    banner_html = f"""
    <div style="
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5em; font-family: 'Arial', sans-serif;">🐟 Cyprus Fish Classifier</h1>
        <p style="font-size: 1.2em; opacity: 0.9; margin-top: 5px;">
            Classification model that allows the identification of fish species, which are living around Cyprus Island 🏝️
        </p>
<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
            
            <a href="https://github.com/{settings.github_repo_id}" target="_blank" style="
                background-color: white;
                color: #333;
                padding: 8px 16px;
                border-radius: 20px;
                text-decoration: none;
                font-weight: bold;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                Source Code
            </a>

            <a href="https://huggingface.co/{settings.huggingface_repo_id}" target="_blank" style="
                background-color: #FFD21E; 
                color: #000;
                padding: 8px 16px;
                border-radius: 20px;
                text-decoration: none;
                font-weight: bold;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <span style="font-size: 1.2em;">🤗</span>
                Hugging Face Model
            </a>

        </div>
    </div>
    """

    # Take test image samples
    valid_examples = [
        f"{settings.image_samples_path}/{ex}"
        for ex in os.listdir(settings.image_samples_path)
    ]

    # --- THEME ---
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", radius_size="md")

    with gr.Blocks(theme=theme, title="Cyprus Fish AI") as demo:
        gr.HTML(banner_html)

        with gr.Row():
            # Left Column
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload or drop an image here",
                    sources=["upload", "clipboard", "webcam"],
                    height=350,
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Drop", variant="secondary")
                    submit_btn = gr.Button("🚀 Analyze", variant="primary")

            # Right Column
            with gr.Column(scale=1):
                output_plot = gr.Label(
                    num_top_classes=5,
                    label="Classification Results",
                )

        gr.Markdown("### 🧪 Try with Examples")
        if valid_examples:
            gr.Examples(
                examples=valid_examples,
                inputs=input_image,
                outputs=output_plot,
                fn=predict_wrapper,
                run_on_click=True,
                cache_examples=False,
            )
        else:
            print("⚠️ No image available.")

        submit_btn.click(predict_wrapper, inputs=input_image, outputs=output_plot)
        input_image.change(predict_wrapper, inputs=input_image, outputs=output_plot)
        clear_btn.click(lambda: None, None, input_image)

    return demo
