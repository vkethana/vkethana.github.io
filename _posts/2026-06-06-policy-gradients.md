---
layout: post
title: "What Policy Gradients Learn on HalfCheetah"
published: false
require-mathjax: true
featured_image: /assets/images/rl/hw2_halfcheetah.jpg
tags:
  - machine learning
  - school project
---
This project implements policy-gradient reinforcement learning from scratch. Unlike imitation
learning, the agent receives no demonstrations: it discovers useful behavior only by sampling
actions, observing rewards, and increasing the probability of actions that led to high returns.

## Comparing the Learned Policies

The three HalfCheetah runs make the effect of variance reduction visible. None learned a clean
running gait, but adding a learned baseline changed the behavior from counterproductive motion
into an awkward strategy that at least moves forward.

| Reward-to-go only | RTG + learned baseline | RTG + tuned baseline |
| :---: | :---: | :---: |
| <video src="/assets/videos/rl/hw2_halfcheetah_rtg_only.mp4" controls autoplay muted loop playsinline width="100%"></video> | <video src="/assets/videos/rl/hw2_halfcheetah_baseline.mp4" controls autoplay muted loop playsinline width="100%"></video> | <video src="/assets/videos/rl/hw2_halfcheetah_policy_gradient.mp4" controls autoplay muted loop playsinline width="100%"></video> |

HalfCheetah is a useful stress test because the policy controls six continuous joints and must
coordinate them over long horizons. The reward encourages forward velocity but does not require
a natural-looking gait, so the baseline-equipped policies exploit an ungainly low crawl.

![HalfCheetah approach comparison](/assets/images/rl/hw2_halfcheetah_approaches.png)

Reward-to-go alone finished with an average evaluation return around `-102`. Adding a learned
baseline raised this to roughly `365`, and tuning the baseline learning rate raised it to roughly
`405`. That is a large quantitative improvement, but the videos show why return alone is an
incomplete description of learned behavior.

## From REINFORCE to Advantage Estimation

The basic estimator is REINFORCE:

$$
\nabla_\theta J(\theta) \approx
\frac{1}{N}\sum_i \sum_t
\nabla_\theta \log \pi_\theta(a_{i,t}\mid s_{i,t}) R_i.
$$

I implemented three common variance-reduction techniques:

* **Reward-to-go:** credit an action only for rewards received after that action.
* **A learned baseline:** subtract a value estimate so the policy learns from relative outcomes.
* **Generalized Advantage Estimation (GAE):** interpolate between low-variance temporal-difference
  estimates and high-variance Monte Carlo estimates.

![Policy-gradient estimator comparison](/assets/images/rl/hw2_cartpole_estimators.png)

The CartPole experiments show how much estimator design matters. Advantage normalization was
especially effective, while reward-to-go alone was not consistently better in this run.

## Choosing the GAE Parameter

GAE computes an exponentially weighted sum of temporal-difference residuals:

$$
\hat A_t^{GAE(\gamma,\lambda)}
= \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}.
$$

![GAE lambda sweep](/assets/images/rl/hw2_gae.png)

On LunarLander, `lambda = 0.98` gave the best result, reaching an evaluation return above 250.
Smaller values introduced too much bias, while values near one inherited more Monte Carlo
variance.

## Takeaway

Policy gradients are conceptually simple, but their raw gradient estimates are noisy. Better
credit assignment substantially improved HalfCheetah's return, yet it still did not produce a
convincing gait. The comparison demonstrates both the value of baselines and the importance of
actually watching an agent instead of judging it only from a curve.
