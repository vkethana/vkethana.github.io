---
layout: post
title: "Fun with Filters and Frequencies"
description: Edge detection, image sharpening, hybrid images, and multiresolution blending for CS 180
published: true
---

# TLDR
I explored fundamental image processing techniques using filters and frequency domain operations. From implementing edge detection from scratch to creating hybrid images and seamlessly blending photos across multiple scales.

<p align="center">
  <img src="/assets/images/frequency/pt2/oraple.jpg" width="400"/>
</p>

# Part 0: Background Information
This is my second project for CS 180 (Intro to Computer Vision) at UC Berkeley. The project explores how filters work in image processing and how we can manipulate images in the frequency domain to achieve interesting effects like edge detection, sharpening, hybrid images, and multiresolution blending.

---

# Part 1: Filters and Edges

## Part 1.1: Finite Difference Operator

I implemented 2D convolution from scratch using only NumPy, then compared it with `scipy.signal.convolve2d`.

### Implementation Details
I created two versions:
- **4-loop version**: Explicitly traverses each kernel element for clarity
- **2-loop version**: Uses vectorized operations with `np.dot` for better performance

Both implementations produce identical results. The key difference from scipy is boundary handling: my implementation uses zero-padding by default, while `scipy.signal.convolve2d` offers various options (fill, wrap, symm).

### Runtime Comparison
The 2-loop implementation is significantly faster than the 4-loop version due to vectorization. However, `scipy.signal.convolve2d` is still much faster than both because it's implemented in optimized C code.

### Results: Finite Difference Filters

I applied finite difference operators Dx = `[1, 0, -1]` and Dy = `[1, 0, -1]^T`:

<p align="center">
  <img src="/assets/images/frequency/pt1/selfie_dx.jpg" width="250"/>
  <img src="/assets/images/frequency/pt1/selfie_dy.jpg" width="250"/>
</p>
<p align="center"><em>Left: Dx (horizontal gradients) | Right: Dy (vertical gradients)</em></p>

I also tested a 9x9 box filter for smoothing:

<p align="center">
  <img src="/assets/images/frequency/pt1/selfie_box.jpg" width="250"/>
  <img src="/assets/images/frequency/pt1/selfie_box_scipy.jpg" width="250"/>
</p>
<p align="center"><em>Left: My implementation (zero-padding) | Right: Scipy (default boundary)</em></p>

---

## Part 1.2: Derivative of Gaussian Filter

Using the cameraman image, I computed partial derivatives and edge detection with finite differences.

<p align="center">
  <img src="/assets/images/frequency/pt1/cameraman_dx.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/cameraman_dy.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/cameraman_gradient_mag.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/cameraman_edge_image.jpg" width="200"/>
</p>
<p align="center"><em>Partial derivatives, gradient magnitude, and binarized edges (threshold=0.20)</em></p>

The edge image shows significant noise. The solution? Apply Gaussian smoothing first.

---

## Part 1.3: Derivative of Gaussian (DoG) Filter

### Method 1: Gaussian → Derivative

First blur the image with a Gaussian, then apply finite difference operators:

<p align="center">
  <img src="/assets/images/frequency/pt1/gauss_cameraman_dx.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/gauss_cameraman_dy.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/gauss_gradient_mag.jpg" width="200"/>
  <img src="/assets/images/frequency/pt1/gauss_edge_image.jpg" width="200"/>
</p>
<p align="center"><em>Gaussian blur then derivatives (threshold=0.10) - much cleaner edges!</em></p>

### Method 2: Single DoG Convolution

By convolution associativity, we can combine the Gaussian and derivative filters first, then apply to the image in one pass.

**The DoG filters:**
<p align="center">
  <img src="/assets/images/frequency/pt1/dog_filters.png" width="600"/>
</p>

**Results:**
<p align="center">
  <img src="/assets/images/frequency/pt1/alternate_gauss_gradient_mag.jpg" width="300"/>
  <img src="/assets/images/frequency/pt1/altnerate_gauss_edge_image.jpg" width="300"/>
</p>
<p align="center"><em>Single-pass DoG results - identical to two-pass method!</em></p>

### Key Insight
Both methods produce virtually identical results, confirming convolution associativity: `(Image ⊗ Gaussian) ⊗ Derivative = Image ⊗ (Gaussian ⊗ Derivative)`. The single-pass method is more computationally efficient.

---

# Part 2: Applications

## Part 2.1: Image "Sharpening"

The unsharp mask filter enhances edges by emphasizing high frequencies. The process:

1. Blur the image to get low frequencies
2. Subtract blurred from original to isolate high frequencies  
3. Add scaled high frequencies back: `sharpened = original + α × (original - blurred)`

### Taj Mahal Results

<p align="center">
  <img src="/assets/images/frequency/input/taj.jpg" width="250"/>
  <img src="/assets/images/frequency/pt2/blurred-taj.jpg" width="250"/>
</p>
<p align="center">
  <img src="/assets/images/frequency/pt2/highfreq-taj.jpg" width="250"/>
  <img src="/assets/images/frequency/pt2/sharpened-taj.jpg" width="250"/>
</p>
<p align="center"><em>Original, blurred, high frequencies, and sharpened (α=2.0)</em></p>

### Varying Alpha Parameter

<p align="center">
  <img src="/assets/images/frequency/pt2/sharpened-alpha0.5-taj.jpg" width="180"/>
  <img src="/assets/images/frequency/pt2/sharpened-alpha1.0-taj.jpg" width="180"/>
  <img src="/assets/images/frequency/pt2/sharpened-alpha2.0-taj.jpg" width="180"/>
  <img src="/assets/images/frequency/pt2/sharpened-alpha5.0-taj.jpg" width="180"/>
</p>
<p align="center"><em>α = 0.5, 1.0, 2.0, 5.0 (more α = more sharpening, but also artifacts)</em></p>

### Additional Example

<p align="center">
  <img src="/assets/images/frequency/input/saint.jpg" width="300"/>
  <img src="/assets/images/frequency/pt2/sharpened-saint.jpg" width="300"/>
</p>
<p align="center"><em>Original and sharpened (α=40.0)</em></p>

### Blur → Sharpen Experiment

Can sharpening recover a blurred image?

<p align="center">
  <img src="/assets/images/frequency/pt2/sharpened-taj.jpg" width="300"/>
  <img src="/assets/images/frequency/pt2/sharpened-blurred-taj.jpg" width="300"/>
</p>
<p align="center"><em>Original sharp vs. blurred then sharpened - some detail lost forever</em></p>

Sharpening helps but can't fully recover lost information. Blurring is irreversible.

---

## Part 2.2: Hybrid Images

Hybrid images combine low frequencies from one image with high frequencies from another. They look different depending on viewing distance!

### Example 1: Tennis Ball + Monster

<p align="center">
  <img src="/assets/images/frequency/input/tennis.jpg" width="250"/>
  <img src="/assets/images/frequency/input/small_monster.jpg" width="250"/>
</p>
<p align="center"><em>Tennis ball (low freq, σ=20) and Monster (high freq, σ=3)</em></p>

**Frequency Analysis:**
<p align="center">
  <img src="/assets/images/frequency/pt2/fft-analysis.jpg" width="700"/>
</p>

**Final Hybrid:**
<p align="center">
  <img src="/assets/images/frequency/pt2/hybrid-tennis-small_monster.jpg" width="400"/>
</p>
<p align="center"><em>Close up: monster. Far away: tennis ball</em></p>

### Example 2: Dog + Cat

<p align="center">
  <img src="/assets/images/frequency/input/dog.jpg" width="200"/>
  <img src="/assets/images/frequency/pt2/hybrid-dog-new_cat.jpg" width="200"/>
  <img src="/assets/images/frequency/input/new_cat.jpg" width="200"/>
</p>

### Example 3: Man + Cat

<p align="center">
  <img src="/assets/images/frequency/input/man.jpg" width="200"/>
  <img src="/assets/images/frequency/pt2/hybrid-man-cat.jpg" width="200"/>
  <img src="/assets/images/frequency/input/cat.jpg" width="200"/>
</p>

---

## Part 2.3: Gaussian and Laplacian Stacks

Gaussian stacks progressively blur images, while Laplacian stacks capture details at each frequency band.

### Apple Stack (for Oraple blending)
<p align="center">
  <img src="/assets/images/frequency/pt2/gaussian-laplacian-stack-orapple-im1.jpg" width="100%"/>
</p>

### Orange Stack (for Oraple blending)
<p align="center">
  <img src="/assets/images/frequency/pt2/gaussian-laplacian-stack-orapple-im2.jpg" width="100%"/>
</p>

---

## Part 2.4: Multiresolution Blending

Using Laplacian stacks, we blend images seamlessly across frequency bands.

### Oraple (Apple + Orange)

**Blending Process:**
<p align="center">
  <img src="/assets/images/frequency/pt2/orapple_blending_process.png" width="100%"/>
</p>
<p align="center"><em>High frequencies show fine details, low frequencies show structure</em></p>

**Final Result:**
<p align="center">
  <img src="/assets/images/frequency/pt2/oraple.jpg" width="400"/>
</p>
<p align="center"><em>The classic Oraple!</em></p>

### Day/Night Blend

<p align="center">
  <img src="/assets/images/frequency/input/day.jpg" width="280"/>
  <img src="/assets/images/frequency/pt2/day_night_blend.jpg" width="280"/>
  <img src="/assets/images/frequency/input/night.jpg" width="280"/>
</p>

### Irregular Mask: Nether Portal

Using a circular mask instead of vertical split:

<p align="center">
  <img src="/assets/images/frequency/input/nether1.jpg" width="300"/>
  <img src="/assets/images/frequency/pt2/circular_mask_blend.jpg" width="300"/>
  <img src="/assets/images/frequency/input/nether2.jpg" width="300"/>
</p>
<p align="center"><em>Circular mask creates a portal effect between dimensions</em></p>

TODO - Add code for how i made the mask
