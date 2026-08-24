# VRSketch2Shape Architecture

This document provides a high-level overview of the `VRSketch2Shape` architecture and its repository structure. `VRSketch2Shape` translates sequential VR 3D sketches into 3D shapes (represented as Signed Distance Functions, or SDFs) using a Latent Diffusion Model conditioned on a Transformer-based sketch encoder.

## System Overview

The system consists of three main neural network components operating together:
1. **Sketch Encoder**: A BERT-based transformer that embeds 3D sequential sketches, capturing the stroke order and spatial points.
2. **VQ-VAE**: A Vector Quantized Variational Autoencoder that compresses high-dimensional 3D shapes (SDF grids) into a compact latent space.
3. **Latent Diffusion Model (LDM)**: A 3D U-Net diffusion model that iteratively denoises a latent code conditioned on the sketch embedding, outputting a generated 3D shape in the latent space which is then decoded by the VQ-VAE.

---

## Directory Structure

```text
VRSketch2Shape/
├── configs/            # YAML configurations for Diffusion and VQ-VAE models
├── dataloader/         # Data loading and preprocessing pipelines
├── docs/               # Documentation
├── eval/               # Evaluation metrics (Chamfer distance, F-score)
├── media/              # Images and videos for README
├── models/             # Neural network definitions and model classes
├── scripts/            # Bash scripts for launching training and inference
├── utils/              # 3D rendering and visualization utilities
├── infer.py            # Main inference and evaluation script
├── environment.yml     # Conda environment definition
└── README.md           # Project documentation
```

---

## Core Components

### 1. Model Definitions (`models/`)

- **`models/sketch2shape_model.py`**: The central model class, `SDFusionSketch2ShapeModel`. It encapsulates the Sketch Encoder, VQ-VAE, and Diffusion Model, handling the forward passes, latent space mapping, and optimization.
- **Sketch Encoder (`BertTokenEncoder`)**: Defined in `sketch2shape_model.py`, this takes sequential 3D points as input. To capture temporal drawing intent, it incorporates:
  - **Positional Embeddings**: To track the order of points within a stroke.
  - **Token Type Embeddings**: To distinguish different strokes and track stroke order.
  - Special tokens (`[SEP]`, `[EOS]`) delineate strokes and the end of the sketch.
- **`models/networks/`**: Contains the lower-level architectures for the `diffusion_networks` (the 3D U-Net) and the `vqvae_networks`.
- **`models/base_model.py`**: Provides a standard interface for PyTorch models, handling checkpoint loading, device placement, and optimizer stepping.

### 2. Data Pipeline (`dataloader/`)

- **`dataloader/sketch_data.py`**: Defines the `Sketch2ShapeDataset`. 
- **Preprocessing**: 
  - Normalizes 3D sketch lines to a unit sphere.
  - Uses `simplify_line_3d` to reduce redundant points while preserving geometry.
- **Sequence Formatting**: Converts variable-length sketches into fixed-length token sequences suitable for BERT. It handles the injection of `[SEP]` and `[EOS]` tokens.
- **Masking Strategy**: Implements random masking of points and entire strokes during training to improve the model's robustness to missing information or varying sketch styles.

### 3. Inference & Evaluation (`infer.py` & `eval/`)

- **`infer.py`**: The entry point for testing. It loads the dataset (either synthetic `syn` or hand-drawn `real` sketches), runs the diffusion sampling process via DDIM (`ddim_steps=100`), and extracts the generated 3D meshes.
- **`eval/eval_obj.py`**: Computes standard 3D reconstruction metrics using PyTorch3D:
  - **Chamfer Distance**: Measures the average distance between points on the generated mesh and the ground truth.
  - **F-score**: Measures precision and recall between the generated and target point clouds.

### 4. Utilities (`utils/`)

- **`utils/util_3d.py`**: Contains helper functions to interface with 3D data, including PyTorch3D mesh renderers and functions for manipulating Signed Distance Functions (SDFs).
- **`utils/visualizer.py`**: Manages the export of generated meshes, images, and logging data during training and inference.

---

## Data Flow (Inference)

1. A 3D sketch (list of strokes containing 3D coordinates) is loaded and preprocessed by `Sketch2ShapeDataset`.
2. The sequence is fed into the `BertTokenEncoder` to produce a conditioned sketch embedding.
3. Random noise is sampled in the latent space of the VQ-VAE.
4. The Latent Diffusion Model (`DiffusionUNet`) iteratively denoises this random noise, conditioned on the sketch embedding, using the DDIM sampler.
5. The denoised latent code is passed through the VQ-VAE decoder to produce a 3D SDF grid.
6. The SDF grid is converted into a 3D mesh (via Marching Cubes) and saved/evaluated.
