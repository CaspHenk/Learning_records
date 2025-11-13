# World Models

**World models**, as their name hints, learn behaviours of the current environment they're in and generate a simulated latent space in such a way that there is no need for real observations anymore, as the model has all the tools to create estimations of what's happening in the real world (apart from actions, which come either from the real world, or from an actor-critic network that generates actions). 


## Components

- **Observation encoder**: the observations are encoded into a latent space using an encoder: $e_t = enc(o_t)$.

- The **model state** is separated in two parts:
    - **The deterministic state $h_t$** contains the long-term sequential information and compresses past information in this compact representation (GRU head). As it uses a Gated Reccurrent Unit network, its gradients stay stable over time, but are also slow to change structure.
    - If we only use $h_t$, we would only have deterministic outputs, resulting in no real estimation. Therefore, we use **the stochastic state $z_t$** to encode information that cannot be encoded deterministically from the past actions (e.g. random environment events, sensory noise, ...). This allows the model to maintain a probabilistic approach to possible states. Since this results in probability distributions (stochastic process), this enables sampling for imagined rollouts, which is needed for policy learning (see Actor-Critic learning).

- **Recurrent state-space model (RSSM)**: this model has three heads:
    - **Sequence model**: estimates the deterministic state $h_t$. It consists of the previously mentioned **GRU**: $h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})$
    - **Transition model**: estimates the **prior** $p(\hat{z}_t | h_t, a_{t-1})$
    - **Inference model**: estimates the **posterior** $q(\hat{z}_t | h_t, e_t)$

- **Decoder**: reconstructs real-world observations based on the model state $\rightarrow dec(z_t, h_t) = \hat{o}_t$

- **Reward predictor**: estimates a reward to be able to give rewards to the rollouts during actor-critic training (usually a simple MLP): $r_{pred}(z_t, h_t) = \hat{r}_t$

- **Continuation predictor**: same principle as reward predictor, but with the continuation variable (if system has to carry on with running or not) (usually a simple MLP): $c_{pred}(z_t, h_t) = \hat{c}_t$

These two do not directly belong to the world model, but are still necessary for our system to be any useful (more details in the actor-critic chapter):

- **Actor policy**: $\pi_\theta(a_t | z_t, h_t)$ predicts an action (or density distributions of actions, depending on the implementation) from a given observation

- **Critic**: $V_\phi(h_t, z_t) = V$ from a given state, estimates a scalar value that represents the expected return of this state (in other words, gives *value* to the current state)

Now how does everything come together?





## Overall training structure

![WM_training](imgs/WM_training.png)

Citing the [Dreamer v3 paper](Dreamer_paper.pdf), "The world model encodes sensory inputs $x_t$ using the encoder (enc) into discrete representations $z_t$ that are predicted by a sequence model with recurrent state $h_t$ given actions $a_t$. The inputs are reconstructed as $\hat{x}_t$ using the decoder (dec) to shape the representations. The actor and critic predict actions $a_t$ and values $v_t$ and learn from trajectories of abstract representations ̂$\hat{z}_t$ and rewards $r_t$ predicted by the world model."

This does not give the training structure of the World model at all. Here's my attempt to make it clearer:

1) Collect observations, actions and rewards from the real world using the current policy (usually a random policy) and store them in a replay buffer. These are data that are going to be used for training, because then we can compare what the world model sees with what actually happened in the real world.

2) Train the world model using the replay buffer:
    - Encode the observations: $enc(o_t) = e_t$;
    - Compute the prior and posterior distributions $p(\hat{z}_t | h_{t-1}, a_{t-1})$ and $q(\hat{z}_t | h_{t-1}, e_t)$ for the stochastic state $z_t$ using the transition and inference models and sample a stochastic state from the **posterior**;
3) Update the deterministic state using the sequence model: $h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$;

4) Predict observations $\hat{o}_t$ decoded from $h_t$ and $z_t$ using the decoder, and using the reward and continuation predictors (usually simple MLPs), also predict $\hat{r}_t$ and $\hat{c}_t$;

5) **Compute losses**: 
    - Reconstruction loss: $L_{recon} = -\log p(o_t | h_t, z_t)$
    - Reward loss: $L_{reward} = MSE(r_t, \hat{r}_t)$
    - Continuation loss: $L_cont = BCE(c_t, \hat{c}_t)$
    - KL loss (name?): $L_{KL} = KL(q(\hat{z}_t | h_t, z_t) || p(\hat{z}_t | h_t, z_t))$

6) **World Model loss**: apply respective coefficients to the different losses: $$L_{world} = \sum_t(\beta_{recon} L_{recon} + \beta_{reward} L_{reward} + \beta_{cont} L_{cont} + \beta_{KL} L_{KL})$$

7) Optimize the encoder, posterior, prior, GRU, decoder and reward predictors using gradient descent (backpropagation).






<!-- :::question

**How does VAE training actually work in the Dreamer case?**

::: -->

:::question

**What's the actual advantage of a world model? Why use this instead of something else?**

It removes the necessity for real observations when training the actor-critic, as the rollouts stay in the latent space. This has two main benefits: first, in a case where it can be hard to gather observations, you can create as many imagined rollouts as you want that can be even as dangerous as you want and that will not affect your real system. The other advantage is that whilst real-data operations can become quite costly, working in the latent space drastically reduces the computational costs of training, making it much faster and cheaper, as the sample efficiency increases drastically with this method.

:::