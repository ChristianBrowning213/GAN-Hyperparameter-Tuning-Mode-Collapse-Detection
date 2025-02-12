# GAN Hyperparameter Tuning & Mode Collapse Detection

This repository showcases an advanced experimental framework for training Generative Adversarial Networks (GANs) with a focus on hyperparameter tuning and detecting mode collapse. It is designed to demonstrate state-of-the-art techniques in GAN architecture, including ResNet-based models, conditional models, skip-connection strategies, and self-attention mechanisms. The project is intended as a portfolio piece to illustrate sophisticated deep learning methodologies rather than as a plug-and-play application.

---

## Overview

In this project, we explore a variety of GAN architectures and training strategies to address two critical challenges in generative modeling:

1. **Hyperparameter Tuning:**  
   The performance of GANs is highly sensitive to hyperparameters such as batch size, learning rate, and the number of training epochs. This framework enables systematic experimentation by allowing quick swaps between different generator and discriminator designs. The goal is to understand how changes in architecture and training configuration affect performance and stability.

2. **Mode Collapse Detection:**  
   Mode collapse is a common issue where the generator produces a limited diversity of outputs despite varying the input noise. To mitigate this, our framework saves generated images at every epoch for multiple classes (using the CIFAR-100 dataset) so that the diversity of outputs can be visually inspected. This continuous monitoring helps in diagnosing and addressing the collapse during training.

---

## Motivation

GANs have revolutionized the field of generative modeling, but their training remains notoriously unstable. The motivation behind this project is to:
- **Investigate Advanced Architectures:** By incorporating state-of-the-art components such as residual blocks, skip connections, and self-attention layers, the project aims to improve the flow of gradients and capture long-range dependencies within generated images.
- **Improve Training Stability:** Techniques like gradient penalty (inspired by WGAN-GP) and spectral normalization are integrated to enforce the Lipschitz constraint on the discriminator and stabilize the training process.
- **Enable In-Depth Analysis:** Detailed logging of loss metrics, model checkpoints, and epoch-wise image outputs allow for a comprehensive analysis of how various architectures and hyperparameters influence the quality and diversity of generated images.

---

## Detailed Explanation: What, How, and Why

### What Is Being Done

The project defines several pairs of generator and discriminator models that can be mixed and matched. Each model incorporates different architectural innovations:
- **Generators**: Designed to produce high-quality, class-conditioned images. Variants include:
  - **ResNetGenerator:** Uses residual blocks to ease the training of deep networks.
  - **SkipConnectionGenerator:** Implements skip connections with a sharpening operation to better preserve image details.
  - **ConditionalGenerator:** Integrates label embeddings to condition image generation on specific classes.
  - **ImprovedGenerator:** Adds self-attention layers to model long-range dependencies and improve global coherence.
  - **ESkipConnectionGenerator:** An enhanced skip connection generator that blends the main pathway and a sharpened skip connection for a balanced output.

- **Discriminators**: Built to differentiate between real and generated images, incorporating class labels to enhance their discriminative power. Variants include:
  - **ResNetDiscriminator:** Uses residual connections to improve gradient flow.
  - **SkipConnectionDiscriminator:** Integrates edge detection within a skip connection to highlight key features.
  - **ConditionalDiscriminator:** Combines image data with label embeddings to evaluate class-specific authenticity.
  - **ImprovedDiscriminator:** Applies spectral normalization and self-attention for robust performance.
  - **ESkipConnectionDiscriminator:** Merges an enhanced skip pathway with the main discriminator network to sharpen decision-making.

### How It Is Being Done

- **Data Preparation:**  
  The CIFAR-100 dataset is used as the benchmark, with images resized and normalized appropriately. This standardization ensures that the models train on consistent data distributions.

- **Model Architecture:**  
  Each generator takes a random noise vector concatenated with a one-hot encoded class label (embedded via an embedding layer) and upsamples this input through a series of transposed convolutions. In parallel, the discriminator concatenates the image with a label embedding and processes the combined tensor through convolutional layers. Advanced layers such as self-attention are interleaved within these architectures to capture spatial correlations across the entire image.

- **Training Loop:**  
  The adversarial training follows a two-step procedure:
  1. **Discriminator Update:**  
     Real images and their corresponding labels are passed through the discriminator, and a loss is computed that penalizes incorrect classifications. Fake images (generated by the current state of the generator) are also evaluated, and a gradient penalty is applied to ensure smooth gradients.
  2. **Generator Update:**  
     The generator is updated based on how well it can fool the discriminator. The goal is to minimize the difference between the discriminator’s predictions for fake images and the label indicating “real.”
     
  Throughout training, key metrics such as the losses for the generator, the discriminator (for both real and fake images), and the combined loss (including the gradient penalty) are logged. Model checkpoints and sample outputs are saved at each epoch.

### Why It Is Being Done

- **Enhancing Model Diversity:**  
  By incorporating multiple advanced architectures, the framework aims to identify designs that produce a diverse set of outputs, thereby combating mode collapse. Visual inspection of generated images helps in evaluating the effectiveness of each architecture.

- **Stabilizing Training:**  
  GAN training can be unstable due to the delicate balance required between the generator and the discriminator. Techniques such as gradient penalty and spectral normalization are critical in maintaining this balance, ensuring that both networks learn effectively without overpowering each other.

- **Facilitating Research and Innovation:**  
  The modular nature of the code allows researchers and practitioners to easily experiment with new ideas. By simply swapping out one architecture for another, one can study the impact of different design choices on the quality and stability of the generated images.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work is a testament to the advancements in GAN research and development. Special thanks to the open-source community and the authors of seminal research papers whose contributions have paved the way for innovative approaches in generative modeling.

---

Happy experimenting and enjoy exploring advanced GAN architectures!


