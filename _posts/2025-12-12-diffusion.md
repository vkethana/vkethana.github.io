---
layout: post
title: "Implementing Diffusion Models"
published: true
tags:
  - machine-learning
  - computer-vision
require-mathjax: true
---

## Part 0: Setup

In this project, I experimented with the DeepFloyd IF diffusion model. This is a text-to-image model that operates in two stages:
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

<p align="center">
  <img src="/assets/images/diffusion/part_1_7_3.png" width="512" title="Text-Guided Edit"/>
</p>

Here, the prompt was "a rainy day". Notice now the images gradually have more and more "rain-like" features. For example, the campanile turns into a bolt of lightning for time step 5 of the first row. Similarly, the happy face drawing starts to show small droplets of rain on its sides.


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
    <figcaption>Flipped: Castle</figcaption>
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
