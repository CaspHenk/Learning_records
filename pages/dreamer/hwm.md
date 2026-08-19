# Hebbian World Model

## Overview

The **Hebbian World Model** (HWM) proposed for this project combines **Hebbian Learning** (see dedicated chapter) with **World Models** (again, see dedicated chapter) - more accurately, the Dreamer architecture. The main idea is to simply replace the weight update with the Dreamer output, and give as input to Dreamer the controller weights, the pre- and post-synaptic activations (and also the environment reward). You can see this input as a sort of image, as it's just three matrices stacked on top of each other, therefore I'll call this representation the **Hebbian image**.


HERE: introduce schematic of how it looks like

## Motivation

<!-- Before reading this section, it is mandatory to have read and understood at least the Hebbian learning and World model chapters. -->

Traditional Hebbian learning setups rely on **Evolutionary Algorithms** (EAs) (see Reinforcement Learning chapter) to evolve synaptic plasticity parameters (these ABCD parameters).

However, as it was explained in the aforementioned chapter, these EAs have several drawbacks:

- They are optimized on the fitness (episodic rewards), making them lose information about fine-grain dynamics during episodes;
- Their sample efficiency is very low, as only one individual in the whole population is kept (*selection* principle)

With the use of Dreamer on top of it, the weight updates receive detailed, intra-episode feedback of the weight dynamics. It transforms the Hebbian learning in a more sample-efficient method for learning using rollouts and Dreamer's replay buffer. We are now able to create a link between the local updates of Hebbian learning and the global reward signal of policy learning and we avoid computing explicit backpropagation on the controller weights (as Dreamer now learns to model the weight dynamics instead of the weights themselves).


## Training pipeline

It is not that different from Dreamer. I'll just add the details that are slightly different. For more details about this part, please look at the Implementation Strategies chapter.

First of all, one important thing is that this pipeline will not use the eventual output images from the environment and will purely base itself on the vector observations (they can contain various elements depending on the environment, but these are basically useful scalars for the controller to understand what's currently going on in the environment). Later on, it could be interesting to think of adding the environment image output, maybe not as an input to the controller but directly to Dreamer, it would break the abstraction but in this way Dreamer would maybe be able to get an understanding of the real world at the same time as of the controller dynamics, eventually making it converge faster, but this is of course only my own speculation...

1. **Replay buffer initialization**: In Dreamer, the replay buffer contains actions, observations and rewards (and also the continuation variable $c_t$). Here, we have to add the Hebbian image and also $\Delta W$, as they are what's going to be used as inputs and outputs to Dreamer. (Theoretically, the actions and observations are not really necessary anymore in the replay buffer as Dreamer will not interact with them anymore)

2. **World model training**: works the same as in the usual Dreamer, the only difference here is obviously the output: Dreamer dreams of controller trajectories in the latent space, predicting how the weights will evolve, and not what actions will come out.

3. **Actor-critic learning**: Works the same as in Dreamer, again just with a different output. 

4. **Controller update**: This is the interesting part: based on the predicted weight updates from Dreamer, the controller updates its weights: 

$$W_{t} = W_{t-1} + \Delta W_{t-1}^{Dreamer} $$




<!-- The **Hebbian World Model** combines both **Hebbian Learning** (see Hebbian Learning page) and the **Dreamer** architecture (see Dreamer page) together for reinforcement learning.
The main idea is simply to let Dreamer model and optimize the synaptic dynamics of the Hebbian controller instead of directly modeling environment transitions.

The main points of this framework include:
- The **Hebbian controller**: interacts with the environment. Gets environment observations as input, produces action(s) as output.
- The **Dreamer world model**: learns to *predict weight updates*, based on the controller's pre- and post-synaptic activations, current weights and reward feedback from environment.
- The **evolutionary algorithm (EA)** previously needed for training

In this framework:
- The **Hebbian controller (HC)** interacts with the environment and produces actions based on its synaptic weights.
- The **Dreamer world model (WM)** learns to *predict weight updates* (`ΔW`) based on the controller’s pre- and post-synaptic activations, current weights, and reward feedback.
- The WM effectively replaces the **evolutionary algorithm (EA)** usually used to train Hebbian controllers, providing a temporally detailed and differentiable learning process. -->


<!-- ## 🎯 Motivation

Traditional Hebbian reinforcement setups rely on **evolutionary algorithms** to evolve synaptic plasticity parameters.  
However, EAs have several drawbacks:
- They only use **episodic rewards**, losing information about intra-episode dynamics.  
- They require **large population sampling**, leading to inefficient training.  
- They lack **temporal credit assignment**, as each episode is evaluated as a single fitness score.

By introducing **Dreamer**, which can imagine latent rollouts and backpropagate gradients through time, we:
- Retain **temporal credit** between local updates and long-term reward.  
- Avoid **explicit backpropagation** through the controller (Dreamer learns a model of the controller instead).  
- Enable **data-efficient learning** through Dreamer’s latent-space imagination and replay buffer.  

--- -->

<!-- ## 🧩 Architecture

### 1. Hebbian Controller (HC)
- Input: Environment observation \( o_t \)
- Output: Action \( a_t \)
- Parameters: Synaptic weights \( W_t \)
- Local activity:
  - Pre-synaptic activations: \( \text{pre}_t \)
  - Post-synaptic activations: \( \text{post}_t \)

The controller acts on the environment using:
\[
a_t = f(o_t, W_t)
\]

### 2. Dreamer World Model (WM)
Instead of modeling environment transitions, the WM models **controller dynamics**:
\[
(\text{pre}_t, \text{post}_t, W_t) \rightarrow \Delta W_t
\]

It learns to predict:
- The **weight update rule**: \( \Delta W_t = g_\phi(\text{pre}_t, \text{post}_t, W_t) \)
- The **reward**: \( \hat{r}_t = r_\psi(W_t) \)
- The **next latent state**: \( z_{t+1} = f_\theta(z_t, \Delta W_t) \)

The WM is trained using Dreamer’s standard objectives:
- **Reconstruction loss** (predicting next latent state)
- **Reward prediction loss**
- **KL regularizer** for stochastic latent space consistency

--- -->

<!-- ## 🔄 Training Pipeline

### Stage 1 — Replay Buffer Initialization

Dreamer requires a replay buffer of transitions.  
Each entry stores:

\[
(\text{pre}_t, \text{post}_t, W_t, \Delta W_t, r_t, \text{next})
\]

#### Initialization options:
1. **Structured Random Updates**  
   Initialize with small random Hebbian-like updates:
   \[
   \Delta W_t = \eta (\alpha \, \text{pre}_t \cdot \text{post}_t^T + \beta \, \text{noise})
   \]
   where \( \alpha, \beta \) are small random coefficients.

2. **Naive Hebbian Rollouts**  
   Run the controller for several episodes using a simple rule (e.g. Oja’s or BCM), collect transitions, and store them in the replay buffer.

--- -->

<!-- ### Stage 2 — Dreamer World Model Training

1. Sample batches from the replay buffer.  
2. Train the RSSM (Recurrent State-Space Model) using:
   - Transition loss between predicted and actual latent states
   - Reward prediction loss
   - KL regularization between prior and posterior latents
3. Dreamer learns to imagine **controller trajectories in latent space**, predicting how weights evolve.

--- -->

<!-- ### Stage 3 — Dreamer-Guided Controller Updates

Once trained, Dreamer can **predict weight updates** based on its learned dynamics:
\[
\Delta W_t^{\text{Dreamer}} = g_\phi(\text{pre}_t, \text{post}_t, W_t)
\]

The controller updates its weights using these predictions:
\[
W_{t+1} = W_t + \Delta W_t^{\text{Dreamer}}
\]

The reward model inside Dreamer is used to evaluate imagined rollouts and optimize \( g_\phi \) toward updates that improve long-term reward.

--- -->

<!-- ### Stage 4 — Iterative Co-Learning

1. HC acts in the environment using Dreamer-predicted updates.  
2. New experience tuples are added to the replay buffer.  
3. Dreamer is periodically retrained on the updated buffer.  

This creates a **closed-loop system**:
> Hebbian activations → Dreamer predicts ΔW → Controller updates → Environment feedback → Replay → Dreamer refinement

--- -->
<!--
## ⚙️ Pseudocode

 ```python
# Initialize controller weights
W = random_weights()

# Initialize replay buffer
replay = []

# Stage 1: Populate replay buffer
for episode in range(N_init):
    for t in range(T):
        pre, post = controller_activations(o_t, W)
        ΔW = naive_hebbian(pre, post) + small_noise()
        a_t = controller_action(o_t, W)
        o_next, r_t = env.step(a_t)
        replay.append((pre, post, W, ΔW, r_t))
        W += ΔW
        o_t = o_next

# Stage 2: Train Dreamer world model
dreamer.train(replay)

# Stage 3: Dreamer-guided learning loop
for episode in range(N_train):
    for t in range(T):
        pre, post = controller_activations(o_t, W)
        ΔW_pred = dreamer.predict_update(pre, post, W)
        a_t = controller_action(o_t, W)
        o_next, r_t = env.step(a_t)
        replay.append((pre, post, W, ΔW_pred, r_t))
        W += ΔW_pred
        o_t = o_next
    dreamer.retrain(replay) -->


:::question

**What's the main issue with RL that I'm trying to solve with this project?**


With Dreamer, the controller's weights are updated via backpropagation, but this causes a huge issue: biological plausibility. Backpropagation through time does not make much sense, because it requires global error signals (and is very different of what happens in a biological context where signals act on certain regions). Furthermore, it updates the synapses globally and does not make use of local information.


:::