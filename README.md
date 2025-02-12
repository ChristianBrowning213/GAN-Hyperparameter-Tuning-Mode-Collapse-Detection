# GAN Hyperparameter Tuning & Mode Collapse Detection

This repository is a portfolio showcase of advanced GAN (Generative Adversarial Network) architectures and training techniques. It features a collection of generators and discriminators—ranging from ResNet-based and conditional models to skip-connection and self-attention enhanced architectures. The primary goal of the project is to experiment with various hyperparameter configurations, detect mode collapse, and generate visual results during training.

> **Note:** Although the code is fully runnable, this project is primarily meant as a demonstration piece rather than a production-ready framework.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Detailed Architecture](#detailed-architecture)
  - [Generator Models](#generator-models)
  - [Discriminator Models](#discriminator-models)
- [Training Procedure](#training-procedure)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Analysis & Visualization](#analysis--visualization)
- [Contributing & Extensions](#contributing--extensions)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project provides an experimental framework for:
- **Hyperparameter Tuning:** Easily adjust parameters like batch size, learning rate, number of epochs, etc.
- **Model Comparison:** Experiment with various GAN architectures including ResNet-based, conditional, skip-connection, and self-attention-enhanced models.
- **Mode Collapse Detection:** Generate and save sample images at each training epoch for several classes (using the CIFAR-100 dataset) so that you can visually inspect if the generator is producing diverse outputs.
- **Logging & Checkpointing:** Automatically log training metrics and save model checkpoints and generated images for further analysis.

---

## Features

- **Multiple GAN Architectures:**  
  The repository includes several generator/discriminator pairs such as:
  - **ResNet-based Models:** Utilize residual blocks to improve gradient flow.
  - **Conditional Models:** Condition image generation on class labels via an embedding layer.
  - **Skip-Connection Models:** Incorporate direct skip connections with additional image sharpening operations.
  - **Improved Models with Self-Attention & Spectral Normalization:** Enhance global context understanding and stabilize training.
  - **Edge-Enhanced Skip Connections:** Leverage edge detection to refine feature extraction in discriminators.

- **Advanced Training Techniques:**  
  - Implements a gradient penalty (inspired by WGAN-GP) to enforce the Lipschitz constraint on the discriminator.
  - Periodically saves generated images, enabling visual monitoring of potential mode collapse.

- **Extensive Logging:**  
  Logs per-epoch metrics (loss values, gradient penalties, etc.) into CSV files. Model checkpoints are saved after every epoch, and example images are stored for later inspection.

---

## Detailed Architecture

### Generator Models

1. **ResNetGenerator:**  
   - Uses residual blocks to enhance feature propagation and model convergence.
   - Concatenates noise with a class label embedding to produce class-conditioned images.

2. **SkipConnectionGenerator:**  
   - Introduces an initial upsampling pathway that acts as a “skip” connection.
   - Applies a sharpening kernel to the skip features to enhance image details.
   - Combines the processed skip connection with the main generation pathway.

3. **ConditionalGenerator:**  
   - Merges noise and label embeddings to condition the output on specific classes.
   - Uses a simple deconvolution architecture with batch normalization and ReLU activations.

4. **ImprovedGenerator:**  
   - Integrates self-attention layers to capture long-range dependencies within generated images.
   - Incorporates progressive upsampling with batch normalization and ReLU activations.

5. **ESkipConnectionGenerator:**  
   - An enhanced skip connection model where the skip pathway is processed via a convolution and sharpening filter.
   - Combines the main and skip outputs using weighted addition to reduce the chance of mode collapse.

### Discriminator Models

1. **ResNetDiscriminator:**  
   - Uses residual connections to improve learning stability.
   - Concatenates a reshaped label embedding with the image data, making it a conditional discriminator.

2. **SkipConnectionDiscriminator:**  
   - Employs a skip connection that is processed with an edge detection kernel.
   - This helps highlight key features that distinguish real from generated images.

3. **ConditionalDiscriminator:**  
   - Merges image data with label embeddings, similar to the ConditionalGenerator.
   - Utilizes several convolutional layers combined with batch normalization and LeakyReLU activations.

4. **ImprovedDiscriminator:**  
   - Incorporates spectral normalization to stabilize the training process.
   - Uses self-attention to better capture spatial dependencies in the image data.

5. **ESkipConnectionDiscriminator:**  
   - An enhanced model that projects the image-label concatenation via a skip connection.
   - Applies edge detection on the skip output before combining it with the main pathway.

---

## Training Procedure

1. **Data Preparation:**  
   - Uses the CIFAR-100 dataset.
   - Images are resized and normalized (mean=0.5, std=0.5 for all RGB channels).

2. **Adversarial Training Loop:**
   - **Discriminator Training:**  
     - Trains on both real images (penalizing low scores) and fake images generated by the generator (penalizing high scores).
     - A gradient penalty is computed to enforce smooth gradients.
   - **Generator Training:**  
     - Updates the generator based on feedback from the discriminator, aiming to generate images that the discriminator classifies as real.
   - **Metrics & Logging:**  
     - Per-epoch loss metrics for the generator and discriminator (including gradient penalty terms) are logged into CSV files.
     - Model checkpoints are saved after every epoch.

3. **Mode Collapse Detection:**  
   - At the end of each epoch, the model generates sample images for a few randomly selected classes.
   - These images are saved in an experiment-specific directory for visual inspection.

---

## Project Structure

