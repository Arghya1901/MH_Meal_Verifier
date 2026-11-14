FROM python:3.11-slim

# System libraries needed by OpenCV and Ultralytics CLIP
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (UID 1000) required by Hugging Face Spaces so SQLite & models can write cleanly
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user

WORKDIR /home/user/app

# Install Python packages (CPU-only PyTorch + requirements + CLIP)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# Copy your project files
COPY --chown=user:user . .

# Switch to UID 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

# Pre-download yoloe-11s-seg.pt AND mobileclip_blt.ts (570 MB) into the server image during build
RUN python -c "from ultralytics import YOLOE; m = YOLOE('yoloe-11s-seg.pt'); pe = m.get_text_pe(['biryani']); m.set_classes(['biryani'], pe)"

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]