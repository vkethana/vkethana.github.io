---
layout: post
title: "Offline Reinforcement Learning for Robot Control"
published: false
require-mathjax: true
featured_image: /assets/images/rl/hw5_cube.jpg
tags:
  - machine learning
  - school project
---
Offline reinforcement learning asks a difficult question: can an agent learn a better policy
from a fixed dataset without collecting any new experience? I implemented and compared SAC+BC,
Implicit Q-Learning (IQL), and Flow Q-Learning (FQL) on OGBench robot-control tasks.

## Learned Cube-Manipulation Policies

All three algorithms learned successful policies from the same fixed dataset:

| SAC+BC | IQL | FQL |
| :---: | :---: | :---: |
| <video src="/assets/videos/rl/hw5_sacbc_cube.mp4" controls autoplay muted loop playsinline width="100%"></video> | <video src="/assets/videos/rl/hw5_iql_cube.mp4" controls autoplay muted loop playsinline width="100%"></video> | <video src="/assets/videos/rl/hw5_fql_cube.mp4" controls autoplay muted loop playsinline width="100%"></video> |

The robot must move the red cube to its target while avoiding distribution shift: actions that
look good to a learned critic may be absent from the dataset and therefore badly estimated.

## Three Ways to Stay Close to the Data

**SAC+BC** combines an actor-critic objective with behavior cloning. The cloning term discourages
unsupported actions, but its weighting parameter is sensitive.

**IQL** avoids querying out-of-distribution actions when learning its value function. It fits an
expectile value estimate, then performs advantage-weighted behavioral cloning.

**FQL** represents the policy as a learned flow from noise to actions. It combines a flow-matching
behavior-cloning objective with Q-guidance toward high-value actions.

## Results

![Cube manipulation success rates](/assets/images/rl/hw5_cube.png)

The best IQL run reached a cube success rate of `1.0`, while the best SAC+BC and FQL runs reached
`0.96`. SAC+BC was more sensitive to its behavior-cloning weight; IQL was comparatively robust
in my sweep.

The navigation task was harder and produced noisier results:

![Ant soccer navigation success rates](/assets/images/rl/hw5_antsoccer.png)

<video src="/assets/videos/rl/hw5_fql_antsoccer.mp4" controls muted loop playsinline width="100%"></video>

## Takeaway

Offline RL is a balancing act between improvement and conservatism. Pure behavior cloning cannot
prefer the best actions in a mixed-quality dataset, but unconstrained value maximization exploits
critic errors. SAC+BC, IQL, and FQL encode three different answers to the same core problem:
improve the policy without leaving the support of the data.
