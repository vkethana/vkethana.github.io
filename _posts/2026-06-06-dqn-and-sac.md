---
layout: post
title: "Deep Q-Learning, Pac-Man, and Soft Actor-Critic"
published: false
require-mathjax: true
featured_image: /assets/images/rl/hw3_mspacman.jpg
tags:
  - machine learning
  - school project
---
This project implements two families of off-policy reinforcement-learning algorithms: Deep
Q-Networks (DQN) for discrete actions and Soft Actor-Critic (SAC) for continuous control.

## A DQN Playing Ms. Pac-Man

The DQN observes stacks of four `84 x 84` grayscale frames and chooses among the Atari action
set. After one million environment steps, the trained policy can play the game directly from
pixels:

<video src="/assets/videos/rl/hw3_mspacman_dqn.mp4" controls autoplay muted loop playsinline width="100%"></video>

DQN learns a neural approximation to the action-value function:

$$
Q(s,a) \leftarrow r + \gamma Q_{\text{target}}(s', \arg\max_{a'}Q(s',a')).
$$

Replay buffers break temporal correlations, while a slowly updated target network prevents the
regression target from changing after every optimizer step. Double DQN further separates action
selection from target evaluation to reduce overestimation.

![DQN results](/assets/images/rl/hw3_dqn.png)

During training, Ms. Pac-Man's epsilon-greedy policy sometimes scores through random actions.
Evaluation uses a greedy policy, so its early return can be lower and less noisy than training
return.

## Soft Actor-Critic on HalfCheetah

SAC extends off-policy learning to continuous action spaces and explicitly rewards entropy. The
resulting policy learns a fast, coordinated HalfCheetah gait:

<video src="/assets/videos/rl/hw3_halfcheetah_sac.mp4" controls autoplay muted loop playsinline width="100%"></video>

The entropy temperature initially fell below `0.02`, then rose and stabilized near `0.125`.
Early in training the random policy already has high entropy, so the optimizer reduces the
temperature. As the policy becomes more deterministic, temperature rises to preserve exploration.

![Automatic temperature tuning](/assets/images/rl/hw3_temperature.png)

## Why Clipped Double-Q Matters

With a single critic, estimation errors are amplified by repeatedly selecting actions with the
largest predicted value. Clipped double-Q trains two critics and backs up the smaller estimate.

<video src="/assets/videos/rl/hw3_hopper_clipq.mp4" controls muted loop playsinline width="100%"></video>

![Single Q versus clipped double Q](/assets/images/rl/hw3_hopper.png)

The single-Q Hopper critic became optimistic without producing a good policy. Clipped double-Q
reached much higher returns because its target values were more conservative.
