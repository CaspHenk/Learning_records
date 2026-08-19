# Hebbian learning

Hebbian learning is based on the use of **Hebbian controllers**, a special type of controller that does not use backpropagation for weight updates but bases itself on the so-called ABCD rules, which are correlation-based weights:

:::theory

$$\Delta w_{ij} = A_{ij} i_i o_j + B_{ij} i_i + C_{ij} o_j + D_{ij}
$$

:::

To "learn", this paradigm makes use of different layers of learning mechanisms:
- **The "inner" loop**: makes use of the aforementioned ABCD rules, which corresponds to online learning during an agent's lifetime (updates the weights of the agent);
- **The "outer" loop**: meta-learning by using an Evolution Algorithm (EA) that evolves the plasticity parameters A,B,C,D;
- **Policy optimization**: From this outer-loop, we identify the emergent behaviour resulting from the interaction between the network and the environment.

## Full training pipeline

Overall, this is how a policy optimization process using Hebbian learning works:

First, define the number of **generations**, which is the number of iterations through the outer loop. Each generation basically creates a population based on the top individuals of the inner loop, but I'll come to it in a few lines. Here's how it works:

1) Initialize a population of N individuals (agents). Each one has random synaptic weights $w_{ij}$, a set of randomly initialized parameters $A,B,C,D$ (and eventually a learning rate $\eta$) and pre-defined structural parameters such as layer size or connection probability;
2) Go through the inner loop for each individual:
  - Reset the environment and neural activations to make it forget what it did with the previous individual;
  - run an episode of $t$ timesteps (continue until $max_{timesteps} is reached, or until time out), which goes as follows:
    - Observe state $s_t$;
    - Compute neuron activations $o_j$
    - Sample an action from the state: $a_t \tilde \pi(s_t)$;
    - Execute $a_t$ and get $r_t$ from the environment, along with $s_{t+1}$
    - Apply the Hebbian update to each weight in the network, following the ABCD rule aforementioned.
  - Compute the **fitness** of the individual: $F_i = \Sigma_t r_t$
3) Compute **fitness** of each individual.

Now we're outside of the inner loop with a bunch of fitness scores that tells us how good each individual is in the population. What happens here is that we only keep the top-performing offspring and discard the rest (**Selection process**). Afterwards, we copy their plasticity parameters (**Reproduction**) and apply some slight perturbation to them (**Mutation**), resulting in newly-formed mutated offspring. Now, we're ready for the next generation with this new population, and this process continues until the desired number of generations is reached. 

As you can see, this method is not very sample-efficient, in the sense that at each generation, a percentage of individuals in the population (which you can determine on your own by the way) is completely deleted and replaced by new individuals. However, this method is still great because it avoids the use of backpropagation (so there's no need for the function to be learned to be derivable! That's a banger).



:::question

 ***Why does a Hebbian controller need a Policy/EA to be able to learn something?***

 The update rule does not use a reward signal to do its update, it's purely based on neuron activities and not on task performance, making it impossible to optimize weights on its own to optimize its behaviour for a given task. Therefore, a Policy/EA is able to apply reward-based optimization on the parameters of the Hebbian controller (see RL Basics chapter).

:::


:::question

***What's the main issue with Hebbian learning?***

:::