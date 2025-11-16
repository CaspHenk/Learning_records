# Actor-Critic Network

## Why actor-critic?

Let's say you have a classical learning algorithm as shown in the RL Basics chapter. In the case your environment is derivable everywhere, well it's quite convenient for your learning pipeline because you can do backpropagation (= gradient descent) using a loss such that the obtained reward is maximized in the current environment. But what about the case where you cannot apply gradient descent because some local derivatives do not exist? You have to find some other ways to be able to get information about how to maximize the reward! Here enters actor-critic learning.

## Actor overview

**In short**: Given an **observation** (synonym for "state") $s$, the goal of the actor network is to find the next **action** (in some cases, for example Dreamer, the probability distribution of actions). To this end, it makes use of a **value function** $V(s)$, that will determine how optimal the current state is (e.g. in a chess game, if your current state gives you Mat in 3, the value function will return a very high value, a higher one than if you had Mat in 5).

## Actor theory

The goal of the **policy gradient method** is to maximize the expected episodic reward. The loss function is the following: $$J(\theta)=\mathbb E_{s\sim d^\pi,a\sim\pi_\theta}  [ A(s,a)],$$

where $A(s,a)$ is the **advantage function** (more details later).

The policy gradient looks like this:

$$\nabla_\theta J(\theta)=\mathbb E _{s,a\sim\pi_\theta} [\nabla_\theta \log\pi_\theta(a|s)\cdot A(s,a)]$$

Let's decompose this equation (I followed [Wikipedia](https://en.wikipedia.org/wiki/Policy_gradient_method)'s description):

- First, $\nabla_\theta \log\pi_\theta(a|s)$ is what we can call the **score function**. Since it's a gradient, you can interpret it as "how changing the parameters $\theta$ would change the probability of taking action $a$ in state $s$ (for now, no notion of reward);
- Second, there's this advantage function, which tells you how good that action was compared to the policy's own expectation for a given state (again, more details in a bit).

The expectation of the combination of the two terms computes an average over all sampled state-action pairs, therefore increasing the probability of actions that turned out better than expected, and reducing the probability of those that turned out worse than expected.

The update rule is called **gradient ascent** (since we maximize) and simply the following: $\theta \leftarrow \theta +\beta\nabla_\theta J(\theta)$.

## Actor derivation

The following part goes a bit more into the details of these calculations.

### REINFORCE

For this method, the advantage function is $A(s,a) = R(\tau)= \sum_{t=0}^{T}\gamma^t r(s_t,a_t)$, the cumulative reward over a certain trajectory $\tau$ that has discrete steps $t \in [0,T]$ ($\gamma$ is the discount factor). The equation for the policy gradient becomes:

$$
\begin{equation}\nabla_\theta J(\theta) = \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t|s_t) \cdot \sum_{t=0}^{T}\gamma^t r(s_t,a_t)\rbrack
\end{equation}$$

### REWARD2GO

Consider Eq.1. Since past rewards do not influence present and future value of the score function, we can use the "causality trick" and change the boundaries for the advantage function sum (the expectation of the product between the score function and past rewards is zero):

$$
\nabla_\theta J(\theta) = \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t|s_t) \cdot \sum_{\mu=t}^{T}\gamma^\mu r(s_\mu,a_\mu)\rbrack$$

Note: $\tau$ represents a certain fixed trajectory. Therefore, $R(\tau)$ is a fixed value. 

This new advantage function results in what they call in the literature the **Q-function**: $\sum_{\mu=t}^{T}\gamma^\mu r(s_\mu,a_\mu) = Q^{\pi_\theta}(s_t,a_t)$.

### Baseline method

REINFORCE and REWARD2GO can lead to high variance in the updates, because the trajectories are sampled from the same policy (**on-policy** methods), therefore the rewards $R(\tau)$ can vary significantly. A good way to reduce variance is to introduce a so-called baseline $b(s)$ (state-dependent function $ \text{States} \rightarrow \mathbb{R}$) to the loss:


$$\nabla_\theta J(\theta) = \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{T}\cdot\nabla_\theta \log\pi_\theta(a_t|s_t) [Q^{\pi_\theta}(s_t,a_t)-b(s)] \rbrack$$

In actor-critic learning, we choose $b(s) = V(s)$, the aforementioned value function. Therefore, we end up with:

$$\nabla_\theta J(\theta) = \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{T}\cdot\nabla_\theta \log\pi_\theta(a_t|s_t) [Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s)] \rbrack$$

Now what do we have so far? We have a pretty well-developed loss function for our actor policy! But the thing is that our advantage function (which is now $A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s)$) has an elegant, optimal form, but we do not have the analytical skills to get these functions easily. Therefore, in the following chapter I'll explain how we can estimate this function.


### The advantage function

There are three methods I'll present to compute the advantage function (spoiler: the value function is not going to be estimated yet, as it is left for the critic, see next chapter):

1. *The Monte Carlo method*

    This is the most straightforward method: just use the cumulative observed reward as an estimate of $Q^{\pi_\theta}(s_t,a_t)$ (simply sample rewards from a trajectory):

    $$Q^{\pi_\theta}(s_t,a_t) = R_t = \sum_{k=t}^{T}\gamma^k r(s_k, a_k)$$

    This is exactly the equality that was stated in the REWARD2GO part. This results in the following advantage function:

    $$A_t(s,a) = R_t - V(s_t)$$


2. *Temporal difference of order 0, TD(0)*

    $$Q^{\pi_\theta}(s_t,a_t) = r_t + \gamma V(s_{t+1})$$

    Therefore, 

    $$ A_t(s,a) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

    We will use the notation $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, as we will need this for the GAE for further calculations.

3. *GAE*

    As shown above, TD only takes one step ahead for approximation, but there can be significant information even further.

    GAE (generalized advantage estimation) includes more steps, using a weighted average (weighted by $\lambda$):

    $$\begin{align*}A_t(s,a) 
    &= \delta_t + \gamma\lambda\delta_{t+1}+{(\gamma\lambda)}^2\delta_{t+2} + ...\\
    &= \sum_{l=0}^\infty (\gamma\lambda)^l\delta_{t+l}
    \end{align*}$$

    Note: the GAE sum stops at the end of the trajectory and obviously does not go to infinity, since we can't really go further.

Now the only thing left is to estimate the value function using the critic network!

## Critic theory

Remember, we have access to trajectories of states, actions and rewards (rollouts). These are used as the training data for the critic. Usually, we'll want for the value function to get as close as possible to the target returns observed during the trajectories.

This time, the loss function looks like that:

$$L_{V}(\phi) = (V_{\phi}(s_t) - R_t)^2,$$

where $\phi$ denotes the model parameters for the critic network.

Here are the methods to estimate $R_t$:

1) Monte-Carlo: $R_t = \sum_{k=t}^{T}\gamma^k r(s_k, a_k)$

2) TD(0): $R_t = r_t + \gamma V(s_{t+1})$

Note: GAE is a method specifically used for advantage function estimation, not simply for estimating the cumulative reward.

In the case of the Dreamer implementation I'm using, GAE is used for the advantage function, with the TD(0) for $R_t$.

## Training loop

1. Retrieve a trajectory $\tau$;

2. Update critic parameters $\phi$:

    - TD: $R_t =  r_t + \gamma V_\phi(s_{t+1})$

    - Compute loss: $L_{V}(\phi) = (V_{\phi}(s_t) - R_t)^2 = (V_{\phi}(s_t) - r_t - \gamma V_\phi(s_{t+1}))^2 $ 

    - Update parameters using gradient descent: $\phi \leftarrow \phi - \alpha \nabla_\phi L_V(\phi) $

    - Retrieve the output of the critic network, $V_\phi (s_t)$

3. Update actor parameters $\theta$:

    - Compute $\delta_t$ using estimated value function: $\delta_t =  r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

    - Compute advantage function using GAE: $ A_t(s,a) = \delta_t + \gamma\lambda\delta_{t+1}+{(\gamma\lambda)}^2\delta_{t+2} + ...$

    - Compute gradient of the loss: $\nabla_\theta J(\theta) \approx \nabla_\theta \log\pi_\theta(a|s)\cdot A_t(s,a)$

    The gradient of the loss is an approximation, as we are replacing the expectation with an estimation of the true value.

    - Update parameters $\theta$ using gradient ascent: $\theta \leftarrow \theta + +\beta\nabla_\theta J(\theta)$

4. Reiterate 1-3 until stop criterion



:::question

**Both the value function and the Q-function are basically approximations of the cumulated observed reward. Why are we computing the difference between both for the advantage function?**

Both are separate problems. For the first one, it is an approximation made to compute the loss function used to optimize the critic network, and the other is for the advantage function. Certainly, they both give the same approximation results, but they have different purposes.

:::


