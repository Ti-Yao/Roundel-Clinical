FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install "pydicom>=2.4" "pylibjpeg>=1.4" "pylibjpeg-libjpeg>=1.4"

RUN pip install --no-cache-dir \
    matplotlib \
    nibabel \
    numpy==1.22.4 \
    pandas \
    Pillow \
    scikit-image==0.20.0 \
    scikit-learn \
    scipy \
    tqdm \
    opencv-python-headless==4.7.0.68 \
    streamlit==1.40.1 \
    streamlit_drawable_canvas \
    imageio


RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
