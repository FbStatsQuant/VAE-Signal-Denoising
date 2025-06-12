# Variational Autoencoder for Signal Denoising

This project implements a **Variational Autoencoder (VAE)** for denoising 1D signals. The model is trained on noisy synthetic signals and learns to reconstruct clean versions using compressed latent representations and reparameterized sampling.

---

## 📌 Highlights

- PyTorch-based VAE architecture
- 1D time series windows (length 50)
- Encoder–Decoder with Gaussian latent sampling
- Reconstructs clean signal from noisy input
- Combines reconstruction loss and KL divergence

---

## 🧠 Model Architecture

- **Input**: Noisy 1D window (length = 50)
- **Latent space**: 2D
- **Architecture**:
  - Encoder → [μ, log(σ²)]
  - Reparameterized sampling:
    
    $z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$
  - Decoder reconstructs the denoised window

- **Loss function**:
  
  $\text{Loss} = \text{MSE}_{\text{reconstruction}} + \text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$

---

## 📈 Output Preview

![vae_reconstruction](vae_reconstruction.png)

---

## ⚙️ Run Instructions

### 1. Install dependencies:
```bash
pip install torch matplotlib numpy
