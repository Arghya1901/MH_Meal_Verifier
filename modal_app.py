from pathlib import Path
import modal

ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi[standard]",
        "uvicorn",
        "python-multipart",
        "ultralytics",
        "pillow",
        "psycopg2-binary",
        "ftfy",
        "regex",
    )
    .pip_install("git+https://github.com/ultralytics/CLIP.git")
    .workdir("/app")
    .run_commands(
        "python -c \"from ultralytics import YOLOE; m = YOLOE('yoloe-11s-seg.pt'); pe = m.get_text_pe(['biryani']); m.set_classes(['biryani'], pe)\""
    )
    .add_local_file(ROOT / "app.py", remote_path="/app/app.py")
    .add_local_file(ROOT / "best.pt", remote_path="/app/best.pt")
    .add_local_file(ROOT / "logo.jpeg", remote_path="/app/logo.jpeg")
    .add_local_dir(ROOT / "static", remote_path="/app/static")
)

app = modal.App("mh-meal-verification")


@app.function(
    image=image,
    cpu=4.0,              # 4 CPU cores (no card required, much faster than 2 cores)
    memory=6144,          # 6 GB RAM
    scaledown_window=120, # stays warm for 2 minutes between clicks
    max_containers=2,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys
    import torch

    torch.set_num_threads(4)

    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    import app as app_module

    # Pre-warm all 104 dish text embeddings once on boot so the first scan doesn't freeze
    if app_module.yoloe_prompt_model is not None:
        try:
            prompts = app_module.build_dish_prompts([])
            with app_module._prompt_lock:
                text_pe = app_module.yoloe_prompt_model.get_text_pe(prompts)
                app_module.yoloe_prompt_model.set_classes(prompts, text_pe)
                app_module._cached_prompt_classes = tuple(prompts)
            print("✅ Pre-warmed 104-dish YOLOE embeddings on 4-core CPU")
        except Exception as e:
            print(f"⚠️ Warmup skipped: {e}")

    return app_module.app