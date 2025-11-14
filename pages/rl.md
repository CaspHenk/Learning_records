# Reinforcement learning Basics

## Policy Learning

![RL_principles](imgs/RL_principles.png)
*Basic policy learning loop*

**Policy learning** aims to adjust parameters $\theta$ of the policy using gradients derived from experience (learning method can vary, the one used for actor-critic learning is described in the corresponding chapter). In math form, it looks like this:

Learn a *policy* $\pi(a|s;\theta)$ is equivalent to maximizing the following objective:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t r_t],$$

where $\gamma$ is the discount factor and $r$ is the reward at time step $t$.

Most of the time, the functions given by the environment return a lot of discontinuities, making it hard for the policy to do backpropagation, as it needs a derivable landscape. Therefore to avoid this problem, a critic is often introduced to estimate a value function $V$ and guide the policy (more details in the actor-critic learning chapter).

One of the main downsides to this method is that it is quite sensitive to local minima and gradient noise.


## Evolutionary Algorithms

**Evolutionary Algorithms (EAs)** have as goal to optimize a **population** of candidate solutions (e.g. neural network weights, control parameters, ...) using parameters inspired by biological evolution to maximize fitness (=cumulative reward over an episode, the lifetime for an individual in the population).

Here, no gradient is needed as this method treats learning as a black-box optimization problem:

$$\max_{\theta} f(\theta),$$

where $f(\theta)$ is the fitness. One notices that only the cumulative reward at the end of the episode is considered, therefore there is no need for gradients, making it practical for discontinuous environment functions, but also that fine-grain information happening at each step is lost.

Some components are important to highlight with this method:

- *Selection*: at the end of a generation, choose the best candidate based on highest fitness value
- *Mutation*: add random noise or changes to candidate parameters
- *Crossover*: Combines parameters from two candidates
- *Replacement*: Forms a new generation based on best individuals


:::question

**What's the reward of an environment, and what's the difference with fitness?**

The **reward** indicates how good a specific action is for a given state and indicates immediate progress towards the environment's objective.

The **fitness** is a different concept used in EAs (described above), and represents the overall performance of an indiviudal over its lifetime, and is computed as the sum (or average) of all rewards obtained during the said lifetime. It measures how good the policy was over its whole lifetime.

:::

:::question

**How does a policy actually impact a controller?**

The goal of a controller on its own (given a certain loss function) is to map its current state to control signals, and using continuous feedback to quantify control error (tracking error, ...). If it's able to backpropagate through its loss function, then great! The optimization of low-level motor dynamics/state-control mappings is possible. However, it does not know which goal to pursue, as the loss function is not reward/objective-based.

For example, imagine you have a robot that has to catch a ball. You can fix optimal joint angles for the robot articulations and using the feedback loop, try to get to this value with the following loss (for each joint):

$$L(\theta) = || \theta_{goal} - \theta||^2$$

The thing is that in this case, $\theta_{goal}$ is fixed, but in such a dynamic problem, this angle could also vary and still be very beneficial for catching that ball. Therefore, the policy adds a level of abstraction and adapts the goal joint angles dynamically based on a reward.

:::

:::question

**What's the difference between Policy learning and Evolutionary Algorithms (EA)?**

Their learning method is fundamentally different:

| Feature | Policy learning | Evolutionary algorithms |
|--------|------------|----------------|
| **Optimization type** | Gradient-based | Gradient-free | 
| **Search space** | Continuous, derivable | Any |
| **Unit of learning** | Single agent updated incrementally | Population of agents evolving separately |
| **Information used** | Rewards + gradients | Rewards only (fitness! no fine-grain info) |
| **Sample efficiency** | High (with replay or model-based) | Low (lots of steps needed to learn for each ep. for each ind.) |
| **Exploration** | Via stochasticity in policy | Via population diversity (mutations) |
| **Scalability** | High for large networks | Computationally expensive |
| **Local vs. Global Search** | Local opt. process (follows grad. of perf. fct.) | Global search (due to variety of individuals) |
| **Applicable to** | Differentiable models | Any model, even rule-based controllers (Hebbian...) |

:::

:::question

**What's the main issue with RL that I'm trying to solve with this project?**

:::