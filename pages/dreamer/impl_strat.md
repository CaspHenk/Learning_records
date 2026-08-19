# Implementation strategies

Please read the Hebbian World models chapter beforehand.

## $\Delta W$ initialization (replay buffer filling)

This step is quite important to test. There are three different ways I thought of doing this:

1) **Random behaviour**: just assign samples of a normal/uniform distribution to $\Delta W$;

2) **Structured random updates**: Follow a randomized rule that resembles the ABCD rule:

$$\Delta W = \eta \, (\alpha \, \text{pre} \cdot \text{post}^T + \beta \cdot \text{noise}),$$

where $\eta$ is the scaling factor, $\alpha , \beta$ are randomly distributed parameters, and pre and post refer to the pre- and post-synaptic activations at the considered time step $t$.

3) **Naive Hebbian rollouts**: try to use some simple Hebbian rule (e.g. Oja's rule) and run the environment for a few steps, and fill the replay buffer accordingly.

:::question

**Why is the scaling factor important?**

It can be seen as the learning rate in classical machine learning. If it's too big, you take giant steps and will probably miss an extremum (and also probably diverge), if it's too small, you'll take forever to learn something. In this case, if the scaling factor is too big for the initialization process, the resulting $\Delta W$ will have too great of an impact on the weights, and if it's too small, the weight updates will have nearly no impact.

:::

