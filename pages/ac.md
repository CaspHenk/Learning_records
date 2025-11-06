# Actor-Critic Network

## Why actor-critic?

Let's say you have a classical learning algorithm (TODO: put a schematic here), with an environment outputting observations, giving you rewards and such (c.f. RL Basics for more). In the case your environment is derivable everywhere, well it's quite convenient for your learning pipeline because you can do backpropagation (= gradient descent) using a loss such that the obtained reward is maximized in the current environment. But what about the case where you cannot apply gradient descent because some local derivatives do not exist? You have to find some other ways to be able to get information about how to maximize the reward! Here enters actor-critic learning.

## Actor

**In short**: Given an **observation** (synonym for "state") $s$, the goal of the actor network is to find the next **action** (in a more precise manner in the case of Dreamer, the probability distribution of actions). To this end, it makes use of a **value function** $V(s)$, that will determine how optimal the current state is (e.g. in a chess game, if your current state gives you Mat in 3, the value function will return a very high value, a higher one than if you had Mat in 5).

:::theory

方策勾配の基本式 : $$J(\theta)=\mathbb E_{s\sim d^\pi,a\sim\pi_\theta}  [ A(s,a)]$$

方策勾配定理により

$$\nabla_\theta J(\theta)=\mathbb E _{s,a\sim\pi_\theta} [\nabla_\theta \log\pi_\theta(a|s)\cdot A(s,a)]$$

Update parameter $\theta$ based on "Policy Gradient" : $\theta \leftarrow \theta +\beta\nabla_\theta J(\theta)$

:::

where $A^{\pi_\theta}(s,a):=Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s)$. This $A$ is called "advantage function". We want to maximize the advantage function.


### Derivation

#### REINFORCE
The equation of Policy Gradient.

$$
\begin{equation}\nabla_\theta J(\theta)=\mathbb E_{\tau \sim p_\theta} \lbrack R(\tau)\cdot\sum_{t=0}^{\infty}\nabla_\theta \log\pi_\theta(a_t|s_t)\rbrack
\end{equation}$$

where $R(\tau)=\sum_{k=0}^{T-1}\gamma^kr(s_k,a_k)$.

※ $\tau$ represents a certain trajectory. Therefore, $R(\tau)$ is a fixed value. 

#### REINFORCE → Reward2GO

Let's split the $R(\tau)$ into two terms.

$$
\begin{align}
R(\tau) &= \sum_{k=0}^{T-1}\gamma^k r(s_k, a_k) \\
        &= \sum_{k=0}^{t-1}\gamma^k r(s_k, a_k)+\sum_{k=t}^{T-1}\gamma^k r(s_k, a_k)
\end{align}
$$

Assume you are at time $t$. 

The first term $[\sum_{k=0}^{t-1}\gamma^k r(s_k, a_k)]$ : The past reward. It vanishes as you take expectation.

The second term $[\sum_{k=t}^{T-1}\gamma^k r(s_k, a_k)]$ : The future reward, which is the reward actually obtained after taking action $a_t$, is equivalent to the Q-function.

Therefore, substitute eq(3) into eq(1):

$$\begin{align}\nabla_\theta J(\theta)&=\mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{\infty}\sum_{k=t}^{T-1}\gamma^k r(s_k, a_k)\cdot\nabla_\theta \log\pi_\theta(a_t|s_t)\rbrack \\
&= \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{\infty}Q^{\pi_\theta}(s_t,a_t)\cdot\nabla_\theta \log\pi_\theta(a_t|s_t)\rbrack


\end{align}$$

ここで問題になるのは、
$Q^{\pi_\theta}(s_t,a_t)$ (将来の報酬の合計)が分散が非常に大きいということ。
これが学習を不安定・非効率にする。

#### Reward2GO → Baseline method

任意の関数 $b(s)$（通常は状態に依存）を引いても勾配の期待値は変わらないことが知られている。

$$\begin{align}\nabla_\theta J(\theta)
&= \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{\infty}[Q^{\pi_\theta}(s_t,a_t)-b(s)]\cdot\nabla_\theta \log\pi_\theta(a_t|s_t)\rbrack


\end{align}$$


Here, we usually choose value function $V(s)$ as $b(s)$. 

Finally, we deriver an Actor-Critic formulation, as mentioned at the top of this page,
$$\begin{align}\nabla_\theta J(\theta)
&= \mathbb E_{\tau \sim p_\theta} \lbrack \sum_{t=0}^{\infty}[Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s)]\cdot\nabla_\theta \log\pi_\theta(a_t|s_t)\rbrack


\end{align}$$




### How to calculate Advantage function


$$A^{\pi}(s,a):=Q^{\pi}(s,a)-V^{\pi}(s)$$

1. Monte Carlo method (モンテカルロ法の時)

    We use the actual cumulative reward observed after taking the action, $R_t$, as an estimate of $Q^{\pi}(s,a)$.

    Therefore, we sample rewards of $t\sim {T-1} $ from a trajectory and get cumulative reward:
    $$R_t=\sum_{k=t}^{T-1}\gamma^k r(s_k, a_k)$$

    Thus,

    $$\therefore A_t(s,a) \approx R_t - V(s_t)$$

2. TD(0)

    We approximate $Q$ as 

    $$Q_t(s,a) \approx r_t + \gamma V(s_{t+1})$$

    Thus, 

    $$\therefore A_t(s,a) \approx r_t + \gamma V(s_{t+1}) - V(s_t)$$

    This is equivalent to TD $\delta_t$. 

    :::note
    理論的には、方策 $\pi$に従うときの Q関数は：

    $$Q^\pi(s,a)=\mathbb E [r_t + \gamma V^\pi(s_{t+1})]$$

    実際の学習中は「期待値」を計算できない。よって、サンプルで近似する。つまり、実際に１ステップ行動したときのサンプル$(s,a,r,s')$に対して、
    $$Q(s,a) \approx r + \gamma V(s_{t+1})$$
    :::

3. GAE

    TD only considers one step ahead for approximation.

    However, we can also include more steps with weighted average. 

    Recall TD is

    $$\delta_t =  r_t + \gamma V(s_{t+1}) - V(s_t)$$

    Then we approximate $A$ as

    $$\begin{align}A_t(s,a) &\approx \delta_t + \gamma\lambda\delta_{t+1}+{(\gamma\lambda)}^2\delta_{t+2}\cdots\\
    &= \sum_{l=0}^\infty (\gamma\lambda)^l\delta_{t+l}
    \end{align}$$

    $\lambda = 0$ →　TD(0)（1ステップ）

    $\lambda = 1$ →　モンテカルロ推定（すべての将来報酬を考慮）

## Critic

As described above, the actor optimization requires a certain value function that's found by the critic network, using a state-action pair as input.

To calculate the advantage function, we need the value finction. 

However, we cannot compute the value function $V(s)$ analitically. 

Therefore, we use the critic network to estimate the expected return for a given state or state-action pair, which is then used to compute the advantage for policy updates.

:::note
The reasons why we need to estimate the value function.

(1) **Exact value functions are unknown**

The true value function $Q^\pi(s,a), V^\pi(s)$ depends on the expectation over all possible future trajectories, which we generally cannot compute analytically in most RL problems.

(2) **Sampling alone is noisy and high variance**

Monte Carlo estimates of returns $R_t$ are unbiased but have high variance.

Using a learned critic to approximate $V(s)$ allows us to reduce variance in the policy gradient.

(3) **Enables TD learning / bootstrapping**

By approximating the critic, we can update it incrementally using TD errors.

:::


### Critic network training

We have access to the full trajectories of states, actions, and rewards. These trajectories serve as the dataset for training the critic network, which is optimized using gradient descent to minimize a specified loss function, typically the mean-squared error between predicted values and target returns.

**Loss function**

$$L_v(\phi) = (V_\phi(s_t)-R_t
)^2$$

1. Monte Carlo: $R_t = \sum_{k=0}^\infty\gamma^kr_{t+k}$

2. n-step: $R_t^{(n)} = \sum_{k=0}^{n-1}\gamma^kr_{t+k}+\gamma^nV(s_{t+n})$

3. TD(0): $R_t = r_t + \gamma V(s_{t+1})$

Then optimize parameter $\phi$ with gradient descent.

## Algorithm: Actor-Critic Training

Actor network : $\pi_\theta$

Critic network : $V_\phi$

### TD(0)

:::theory

TD(0)

1. $\pi_\theta$でtrajectoryの生成 : $\tau$

2. Update Critic parameter $\phi$

    TD :   $\delta_t =  r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

    Loss : $L(\phi)= \delta_t^2$

    Update $\phi \leftarrow \phi - \alpha \nabla_\phi L(\phi) $

3. Update Actor prameter $\theta$

    Advantage function (TD(0)) : $ A_t(s,a)=  r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

    Calculate gradient : $\nabla_\theta J(\theta)\approx\nabla_\theta \log\pi_\theta(a|s)\cdot A_t(s,a)$

    Update $\theta \leftarrow \theta +\beta\nabla_\theta J(\theta)$


4. Iteration 1~3
:::


### GAE
:::theory
GAE

1. $\pi_\theta$でtrajectoryの生成 : $\tau$

2. Update Critic parameter $\phi$

    TD :   $\delta_t =  r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

    Loss : $L(\phi)= \delta_t^2$

    Update $\phi \leftarrow \phi - \alpha \nabla_\phi L(\phi) $

3. Update Actor prameter $\theta$

    TD :   $\delta_t =  r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

    Advantage function (GAE) : $ A_t(s,a) \approx \delta_t + \gamma\lambda\delta_{t+1}+{(\gamma\lambda)}^2\delta_{t+2}\cdots$

    Calculate gradient : $\nabla_\theta J(\theta)\approx\nabla_\theta \log\pi_\theta(a|s)\cdot A_t(s,a)$

    Update $\theta \leftarrow \theta +\beta\nabla_\theta J(\theta)$


4. Iteration 1~3
:::


