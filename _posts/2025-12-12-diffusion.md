---
layout: post
title: "Diffusion Image Generation Models from Scratch in PyTorch"
published: true
tags:
  - machine-learning
  - computer-vision
require-mathjax: true
toc:
  - name: Part 0 - Setup
    link: "#part-0-setup"
  - name: Part 1 - Sampling Loops
    link: "#part-1-sampling-loops"
    subsections:
      - name: 1.1 Forward Process
        link: "#11-implementing-the-forward-process"
      - name: 1.2 Classical Denoising
        link: "#12-classical-denoising"
      - name: 1.3 One-Step Denoising
        link: "#13-one-step-denoising"
      - name: 1.4 Iterative Denoising
        link: "#14-iterative-denoising"
      - name: 1.5 Diffusion Sampling
        link: "#15-diffusion-model-sampling"
      - name: 1.6 CFG
        link: "#16-classifier-free-guidance-cfg"
      - name: 1.7 Image-to-Image
        link: "#17-image-to-image-translation"
      - name: 1.8 Visual Anagrams
        link: "#18-visual-anagrams"
      - name: 1.9 Hybrid Images
        link: "#19-hybrid-images"
  - name: Part B - Flow Matching
    link: "#part-b-flow-matching-from-scratch"
    subsections:
      - name: 1. Single-Step Denoising
        link: "#1-single-step-denoising-unet"
      - name: 2. Training Diffusion Model
        link: "#2-training-a-diffusion-model"
---

## Table of Contents
- [Part 0: Setup](#part-0-setup)
- [Part A: Sampling Loops](#part-1-sampling-loops)
  - [1.1 Implementing the Forward Process](#11-implementing-the-forward-process)
  - [1.2 Classical Denoising](#12-classical-denoising)
  - [1.3 One-Step Denoising](#13-one-step-denoising)
  - [1.4 Iterative Denoising](#14-iterative-denoising)
  - [1.5 Diffusion Model Sampling](#15-diffusion-model-sampling)
  - [1.6 Classifier-Free Guidance (CFG)](#16-classifier-free-guidance-cfg)
  - [1.7 Image-to-Image Translation](#17-image-to-image-translation)
  - [1.8 Visual Anagrams](#18-visual-anagrams)
  - [1.9 Hybrid Images](#19-hybrid-images)
- [Part B: Flow Matching from Scratch!](#part-b-flow-matching-from-scratch)
  - [1. Single-Step Denoising UNet](#1-single-step-denoising-unet)
  - [2. Training a Diffusion Model](#2-training-a-diffusion-model)

---

## TLDR

In this project, I explore techniques for sampling from pretrained diffusion models [(Part A)](#part-1-sampling-loops) and train my own class-conditioned flow matching model from scratch using a UNet [(Part B)](#part-b-flow-matching-from-scratch).

Here's an example of an image sampled from the model I trained in part B, which can generate images of any handwritten digit between 0 and 9:
<p align="center">
  <img src="/assets/images/diffusion/diffusion.gif" alt="An image being generated from scratch" />
</p>

## Part 0: Setup

In this part, I experimented with the DeepFloyd IF diffusion model. This is a text-to-image model that operates in two stages:
1.  **Stage 1**: Generates a 64x64 resolution image from the text prompt.
2.  **Stage 2**: Upscales the image to 256x256 and adds details.

I generated images for three different prompts using 20 inference steps. The random seed used for *all parts* of this project is **100**.

### "An oil painting of a snowy mountain village"
<p align="center">
  <img src="/assets/images/diffusion/part0_stage1_castle_20_inference_steps.png" width="200" title="Stage 1"/>
  <img src="/assets/images/diffusion/part0_stage2_castle_20_inference_steps.png" width="400" title="Stage 2"/>
</p>

### "A photo of a cat"
<p align="center">
  <img src="/assets/images/diffusion/part0_stage1_cat_20_inference_steps.png" width="200" title="Stage 1"/>
  <img src="/assets/images/diffusion/part0_stage2_cat_20_inference_steps.png" width="400" title="Stage 2"/>
</p>

### "A photo of a temple"
<p align="center">
  <img src="/assets/images/diffusion/part0_stage1_temple_20_inference_steps.png" width="200" title="Stage 1 (20 steps)"/>
  <img src="/assets/images/diffusion/part0_stage2_temple_20_inference_steps.png" width="400" title="Stage 2 (20 steps)"/>
</p>

The outputs here are pretty good in my opinion, which is a testament to the quality of Deepfloyd IF's training process.

#### Comparison with More Inference Steps
I also generated the temple image with 100 inference steps. This lets us see if the quality improves with more denoising iterations.
<p align="center">
  <img src="/assets/images/diffusion/part0_stage1_temple_100_inference_steps.png" width="200" title="Stage 1 (100 steps)"/>
  <img src="/assets/images/diffusion/part0_stage2_temple_100_inference_steps.png" width="400" title="Stage 2 (100 steps)"/>
</p>

The image turned out a little bit oversaturated, which makes me believe that more inference steps were not necessary, at least for this specific prompt. 
That said, it didn't ruin the image either, so my conclusion is that the number of inference steps has to be chosen qualitatively and depends on the prompt and model being used.

---

## Part 1: Sampling Loops

### 1.1 Implementing the Forward Process

The forward process in diffusion models adds noise to a clean image $x_0$ to produce a noisy image $x_t$ at timestep $t$. The process is defined by the equation:

$$
q(x_t | x_0) = \mathcal{N}(x_t ; \sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t)\mathbf{I})
$$

Which allows us to sample $x_t$ directly:

$$
x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

Here are the results of the forward process on the Campanile image at different noise levels ($t \in \{250, 500, 750\}$):

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/campanile.png" width="150"/>
    <figcaption>Original</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_1_campanile_250.png" width="150"/>
    <figcaption>t=250</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_1_campanile_500.png" width="150"/>
    <figcaption>t=500</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_1_campanile_750.png" width="150"/>
    <figcaption>t=750</figcaption>
  </figure>
</p>

### 1.2 Classical Denoising

I first attempted to remove the noise using classical Gaussian blurring. As expected, this simple technique fails to recover the details, blurring out both the noise and the high-frequency content of the image.

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_2_campanile_250_blurred.png" width="150"/>
    <figcaption>t=250 Blurred</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_2_campanile_500_blurred.png" width="150"/>
    <figcaption>t=500 Blurred</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_2_campanile_750_blurred.png" width="150"/>
    <figcaption>t=750 Blurred</figcaption>
  </figure>
</p>

No amount of Gaussian blurring can bring back parts of the image that were already lost. We need something more sophisticated.

### 1.3 One-Step Denoising

Using a pretrained diffusion model, we can try to recover $x_0$ in a single step. The model is trained to estimate the noise $\epsilon$ in a noisy image $x_t$. Given the estimate $\epsilon_\theta(x_t, t)$, we can approximate $x_0$ by inverting the forward process equation:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}
$$

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_denoise_250.png" width="150"/>
    <figcaption>Denoised (t=250)</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_denoise_500.png" width="150"/>
    <figcaption>Denoised (t=500)</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_denoise_750.png" width="150"/>
    <figcaption>Denoised (t=750)</figcaption>
  </figure>
</p>

The model does a much better job than Gaussian blur, but for high noise levels (t=750), the one-step reconstruction is blurry and lacks fine detail. This is because the initial assumption of mapping directly to $x_0$ is difficult when the signal is heavily corrupted.

### 1.4 Iterative Denoising

To get high-quality images, we denoise iteratively. Starting from pure noise or a noisy image, we repeatedly apply the update step:

$$
x_{t'} = \frac{\sqrt{\bar\alpha_{t'}}\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t'})}{1 - \bar\alpha_t} x_t + v_\sigma
$$

which effectively steps from $t$ to $t'$ by removing a fraction of the predicted noise and adding new variance. In practice, we don't iteratively denoise over all 1000 timesteps, as that would be too costly. Instead, we iterate over a strided subset of timesteps (for this assignment, I set the stride to 30)

Here is the progression of iterative denoising (using strided sampling):

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_690_denoise.png" width="120" title="t=690"/>
    <figcaption>t=690</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_540_denoise.png" width="120" title="t=540"/>
    <figcaption>t=540</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_390_denoise.png" width="120" title="t=390"/>
    <figcaption>t=390</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_240_denoise.png" width="120" title="t=240"/>
    <figcaption>t=240</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_90_denoise.png" width="120" title="t=90"/>
    <figcaption>t=90</figcaption>
  </figure>
</p>

**Comparison:**
<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_gaussian_blurred.png" width="150"/>
    <figcaption>Gaussian Blur</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_non_iterative_denoise.png" width="150"/>
    <figcaption>One-Step</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_3_step_final_denoise.png" width="150"/>
    <figcaption>Iterative</figcaption>
  </figure>
</p>

The iterative result is significantly sharper than the one-step estimation.

### 1.5 Diffusion Model Sampling

We can generate new images by running the iterative denoising loop starting from pure Gaussian noise ($x_T \sim \mathcal{N}(0, \mathbf{I})$) with the prompt "a high quality photo".

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_5_sample1.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_5_sample2.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_5_sample3.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_5_sample4.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_5_sample5.png" width="128"/>
  </figure>
</p>

### 1.6 Classifier-Free Guidance (CFG)

To improve image quality and prompt adherence, I implemented Classifier-Free Guidance, also known as CFG. We compute two noise estimates: one conditional on the text prompt ($\epsilon_{cond}$) and one unconditional ($\epsilon_{uncond}$). The final noise estimate is:

$$
\epsilon = \epsilon_{uncond} + \gamma (\epsilon_{cond} - \epsilon_{uncond})
$$

where $\gamma > 1$ is the guidance scale. This pushes the image towards the prompt.

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_6_sample1.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_6_sample2.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_6_sample3.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_6_sample4.png" width="128"/>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_6_sample5.png" width="128"/>
  </figure>
</p>

The generated images are sharper and more clearly defined compared to the unguided samples.

### 1.7 Image-to-Image Translation
By taking a real image, adding noise to it up to a certain timestep $t$, and then running the iterative denoising process from there, we can edit images. This allows us to balance maintaining the original structure (via the starting noisy image) and generating new details (via the denoising loop).

Below I show edits of the Campanile image at noise levels [1, 3, 5, 7, 10, 20] with the conditional text prompt "a high quality photo".
I also show two edits of my own test images, which are captioned "original" in the below image:

<p align="center">
  <img src="/assets/images/diffusion/part1_7_im2_im.png" width="512" title="Original/Mask"/>
</p>
#### 1.7.1 SDEdit

**Artificial Image Transition (SDEdit):**
<p align="center">
  <img src="/assets/images/diffusion/part1_7_1_artificial_img.png" width="150" title="Original"/>
  <img src="/assets/images/diffusion/part1_7_1_artificial_img_transition.png" width="500" title="Transition"/>
</p>

**Hand-Drawn Image Transition:**
<p align="center">
  <img src="/assets/images/diffusion/part1_7_1_drawn1.png" width="150" title="Sketch 1"/>
  <img src="/assets/images/diffusion/part1_7_1_drawn1_transition.png" width="500" title="Transition 1"/>
</p>
<p align="center">
  <img src="/assets/images/diffusion/part1_7_1_drawn2.png" width="150" title="Sketch 2"/>
  <img src="/assets/images/diffusion/part1_7_1_drawn2_transition.png" width="500" title="Transition 2"/>
</p>

#### 1.7.2 Inpainting
We can use a mask to keep parts of the image constant while denoising the rest. At each step of the backward process, we force the pixels outside the mask to match the noisy version of the original image, while letting the model hallucinate content inside the mask.

<p align="center">
  <img src="/assets/images/diffusion/part1_7_2.png" width="512" title="Inpainted Result"/>
</p>

#### 1.7.3 Text-Conditional Image-to-Image Translation
We can guide the SDEdit process with specific text prompts to change the style or content of the image.
Results for all three images are below:

Here, the prompt was "a rainy day". Notice now the images gradually have more and more "rain-like" features. For example, the campanile turns into a bolt of lightning for time step 5 of the first row. Similarly, the happy face drawing starts to show small droplets of rain on its sides.

<p align="center">
  <img src="/assets/images/diffusion/part_1_7_3.png" width="512" title="Text-Guided Edit"/>
</p>

### 1.8 Visual Anagrams

Visual anagrams are images that look like one thing when upright and another when flipped. I implemented this by averaging the noise estimates for two different prompts, one computed on the upright image and one on the flipped image:

$$
\epsilon_{final} = \frac{1}{2} (\epsilon_\theta(x_t, t, p_1) + \text{flip}(\epsilon_\theta(\text{flip}(x_t), t, p_2)))
$$

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_8_castle_skull.png" width="200"/>
    <figcaption>Upright: Castle/Village</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_8_castle_skull_flipped.png" width="200"/>
    <figcaption>Flipped: Skull</figcaption>
  </figure>
</p>

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_8_citadel_castle.png" width="200"/>
    <figcaption>Upright: Citadel</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_8_citadel_castle_flipped.png" width="200"/>
    <figcaption>Flipped: Temple</figcaption>
  </figure>
</p>

### 1.9 Hybrid Images

Hybrid images combine the low frequencies of one image with the high frequencies of another. We can generate these with diffusion by combining noise estimates:

$$
\epsilon_{final} = f_{low}(\epsilon_\theta(x_t, t, p_1)) + f_{high}(\epsilon_\theta(x_t, t, p_2))
$$

<p align="center">
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_9_hybrid1.png" width="200"/>
    <figcaption>Hybrid Image 1: Rainy Day + Photo of Dog</figcaption>
  </figure>
  <figure style="display:inline-block; margin:5px;">
    <img src="/assets/images/diffusion/part1_9_hybrid2.png" width="200"/>
    <figcaption>Hybrid Image 2: A lithograph of a skull + photo of a castle</figcaption>
  </figure>
</p>

---

# Part B: Flow Matching from Scratch!

In this part, we implement a diffusion model from scratch using the MNIST dataset. We start with a simple single-step denoiser and then move on to a full diffusion model with time conditioning. Unless otherwise stated, the random seed used for all subparts was **100**.

## 1. Single-Step Denoising UNet

### 1.1 Architecture
The backbone of our denoiser is a UNet. At its core, a UNet is just an autoencoder with a twist: it compresses the image into a bottleneck to capture global context (like "this is a digit 8") and then expands it back to the original size. The "twist" is the skip connections—wires that bypass the bottleneck and plug the detailed, high-resolution features from the encoder directly into the decoder. This lets the network reconstruct fine details (like edges and noise) that would otherwise be lost in compression.

<p align="center">
  <img src="/assets/images/diffusion/part2/unconditional_arch.png" width="700" title="UNet Architecture"/>
</p>

### 1.2 Noising Process Visualization
The noising process adds Gaussian noise to a clean image $x$.
$$ z = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

Here is the effect of varying $\sigma$ on a clean image:

<p align="center">
  <img src="/assets/images/diffusion/part2/part_1_2_noising_process.png" width="661" title="Noising Process"/>
</p>

### 1.2.1 Training the Denoiser
I trained a UNet to denoise images with $\sigma = 0.5$. The objective is to minimize the L2 distance between the denoised image and the original clean image:
$$ L = \mathbb{E}_{z,x} \|D_{\theta}(z) - x\|^2 $$

**Training Loss Curve:**
<p align="center">
  <img src="/assets/images/diffusion/part2/part_1_2_1_plot.png" width="567" title="Training Loss"/>
</p>

**Denoising Results (Epoch 1 vs Epoch 5):**
The model learns to remove the noise effectively after just a few epochs.
<p align="center">
  <figure style="display:block; margin:40px auto;">
    <img src="/assets/images/diffusion/part2/part_1_2_epoch1_denoise_results.png" width="800"/>
    <figcaption>Epoch 1</figcaption>
  </figure>
  <figure style="display:block; margin:40px auto;">
    <img src="/assets/images/diffusion/part2/part_1_2_epoch5_denoise_results.png" width="800"/>
    <figcaption>Epoch 5</figcaption>
  </figure>
</p>

### 1.2.2 Out-of-Distribution Testing
The model was trained only on $\sigma=0.5$. Here I tested it on other noise levels. It performs reasonably well on lower noise levels but struggles when the noise level is much higher than what it was trained on (e.g., $\sigma=1.0$).

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.png" width="300" />
        $\sigma=0.0$
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.2.png" width="300" />
        $\sigma=0.2$
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.4.png" width="300" />
        $\sigma=0.4$
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.5.png" width="300" />
        $\sigma=0.5$
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.6.png" width="300" />
        $\sigma=0.6$
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_0.8.png" width="300" />
        $\sigma=0.8$
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_1_2_ood_sigma_1.0.png" width="300" />
        $\sigma=1.0$
      </td>
      <td></td> 
    </tr>
  </table>
</div>

### 1.2.3 Denoising Pure Noise
Here, I trained the model to denoise pure noise (i.e., mapping $\mathcal{N}(0, I)$ to MNIST digits).

<p align="center">
  <img src="/assets/images/diffusion/part2/part_1_2_3_pure_noise_loss_curve.png" width="567" title="Loss Curve"/>
</p>

<p align="center">
  <img src="/assets/images/diffusion/part2/part_1_2_3_generate_from_pure_noise.png" width="800" title="Results"/>
</p>

Interestingly, the model manages to generate digit-like shapes, but they are often blurry or hybrids of multiple digits. 
This is because the mapping from pure noise to a specific digit is one-to-many and highly ambiguous, so the L2 loss forces the model to output the "average" of all possible digits, resulting in blurry blobs.

---

## 2. Training a Diffusion Model

Now we move to a proper diffusion model (Time-Conditioned UNet), where we iteratively denoise the image.

### 2.1 Adding Time Conditioning
To perform iterative denoising, the model needs to know the current noise level (or timestep $t$). We inject this information into the UNet using fully connected blocks (FCBlocks).

<p align="center">
  <img src="/assets/images/diffusion/part2/conditional_arch_fm.png" width="600" title="Conditioned UNet"/>
</p>

The scalar $t$ is fed into two fully connected blocks (`fc1_t`, `fc2_t`) to produce scaling coefficients. These coefficients are then used to modulate the feature maps at specific points in the UNet.

Specifically, $t$ is used to scale the activations after the unflatten step ($t_1$) and after the first upsampling block ($t_2$):

```python
# fc1_t and fc2_t are small MLPs that project the scalar t to channel dimensions
t1 = fc1_t(t)
t2 = fc2_t(t)

# Modulate the unflattened features
unflatten = unflatten * t1

# ... intermediate layers ...

# Modulate the first upsampling block
up1 = up1 * t2
```

Training involves picking a random image $x_1$, a random timestep $t$, adding noise to get $x_t$, and training the network.
The loss at every time step is calculated based on how well the model prediction conditioned on noisy image $x_t$ and time step $t$ matches $x_1 - x_0$: the clean image *minus* the random noise.

<p align="center">
  <img src="/assets/images/diffusion/part2/algo1_t_only_fm.png" width="600" title="Training Algorithm"/>
</p>

### 2.2 Time-Conditioned UNet Training
I trained the UNet conditioned on the timestep $t$.

<p align="center">
  <img src="/assets/images/diffusion/part2/part2_2_time_conditioned_unet_plot.png" width="567" title="Time-Conditioned Loss"/>
</p>

### 2.3 Time-Conditioned Sampling
Sampling starts from pure noise $x_0 \sim \mathcal{N}(0, 1)$ and iteratively refines it to a clean image $x_1$.

<p align="center">
  <img src="/assets/images/diffusion/part2/algo2_t_only_fm.png" width="800" title="Sampling Algorithm"/>
</p>

Here are the sampling results at different epochs and different seeds (100, 101, and 102):

<p align="center">
  <img src="/assets/images/diffusion/part2/part2_3.png" width="717" title="Sampling Results"/>
</p>

<p align="center">
  <img src="/assets/images/diffusion/part2/part2_3_ex2.png" width="717" title="Sampling Results"/>
</p>

<p align="center">
  <img src="/assets/images/diffusion/part2/part2_3_ex3.png" width="717" title="Sampling Results"/>
</p>

### 2.4 Adding Class-Conditioning to UNet

To improve the generation quality and gain control over the output, we condition the UNet on both the timestep $t$ and the digit class $c$. This allows us to ask the model for a "5" or a "7" specifically.

#### Architectural Changes
Similar to time conditioning, we inject the class information $c$ (a one-hot vector) into the network. We add two more FCBlocks (`fc1_c`, `fc2_c`) to process the class vector.

The class conditioning is added to the time conditioning, meaning the modulation signal becomes a combination of both:

```python
# c is a one-hot vector for the digit class
c1 = fc1_c(c)
c2 = fc2_c(c)

# Combine with time embedding and modulate
unflatten = (c1 * unflatten) + t1
# ...
up1 = (c2 * up1) + t2
```

We also use dropout on the class conditioning (setting it to a null token with $p=0.1$) to enable Classifier-Free Guidance later.

<p align="center">
  <img src="/assets/images/diffusion/part2/algo3_c_fm.png" width="600" title="Class-Conditioned Training"/>
</p>

### 2.5 Training the UNet

We train the class-conditioned UNet using the same process as before, but with the added class labels.

<p align="center">
  <img src="/assets/images/diffusion/part2/part_2_5_lr_sch_loss_curve.png" width="576" title="Class-Conditioned Loss"/>
</p>

### 2.6 Sampling from the UNet

We use Classifier-Free Guidance (CFG) during sampling to improve quality. The final noise estimate is a combination of the conditional and unconditional estimates:
$$ \epsilon = \epsilon_{uncond} + \gamma (\epsilon_{cond} - \epsilon_{uncond}) $$

<p align="center">
  <img src="/assets/images/diffusion/part2/algo4_c_fm.png" width="600" title="CFG Sampling"/>
</p>

By guiding the model with class labels, we can generate specific digits. Here are the results over 10 epochs using Classifier-Free Guidance ($\gamma=5.0$).

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_with_lr_sch_epoch1.png" width="220" />
        <br>
        Epoch 1
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_with_lr_sch_epoch5.png" width="220" />
        <br>
        Epoch 5
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_with_lr_sch_epoch10.png" width="220" />
        <br>
        Epoch 10
      </td>
    </tr>
  </table>
</div>

#### Can we get rid of the annoying learning rate scheduler?

I tried training the model with a constant learning rate of 1e-4 instead of using an exponential decay scheduler. To account for the fact that the learning rate no longer decreases, I used AdamW with a weight decay of **1e-4**. As shown in the loss curve, the training was still stable.

<p align="center">
  <img src="/assets/images/diffusion/part2/part_2_5_constant_lr_loss_curve.png" width="567" title="Constant LR Loss"/>
</p>

The sampling results are also comparable to the scheduled version, suggesting that for this specific task and architecture, a well-tuned constant learning rate is sufficient.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_constant_lr_epoch1.png" width="220" />
        <br>
        Epoch 1
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_constant_lr_epoch5.png" width="220" />
        <br>
        Epoch 5
      </td>
      <td align="center">
        <img src="/assets/images/diffusion/part2/part_2_6_constant_lr_epoch10.png" width="220" />
        <br>
        Epoch 10
      </td>
    </tr>
  </table>
</div>
