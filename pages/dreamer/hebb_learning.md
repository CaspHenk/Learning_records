# Hebbian learning

Hebbian learning is based on the use of **Hebbian controllers**, a special type of controller that does not use backpropagation for weight updates but bases itself on the so-called ABCD rules, which are correlation-based weights:

:::theory

$$\Delta w_{ij} = A_{ij} i_i o_j + B_{ij} i_i + C_{ij} o_j + D_{ij}
$$

:::


:::question

 ***Why does a Hebbian controller need a Policy/EA to be able to learn something?***

 The update rule does not use a reward signal to do its update, it's purely based on neuron activities and not on task performance, making it impossible to optimize weights on its own to optimize its behaviour for a given task. Therefore, a Policy/EA is able to apply reward-based optimization on the parameters of the Hebbian controller (see RL Basics chapter).

:::


:::question

***What's the main issue with Hebbian learning?***

:::


<!-- # Evolutionary Hebbian Learning Pipeline (ABCD Rule + Policy)

This document describes the full training pipeline of a **Hebbian learning scheme** using the **ABCD rule** combined with an **Evolutionary Algorithm (EA)** and a **policy network**.

---

## 🧠 Overview

This approach merges **three learning mechanisms**:

1. **Hebbian plasticity (ABCD rule)** – local, online learning during an agent’s lifetime.
2. **Evolutionary algorithm (EA)** – outer-loop meta-learning that evolves plasticity parameters.
3. **Policy optimization** – the emergent behavior resulting from the interaction of the network and environment.

---

## ⚙️ Learning Levels

| Level | Mechanism | What it learns | Time scale |
|--------|------------|----------------|-------------|
| **Inner loop (lifetime)** | Hebbian ABCD rule | Synaptic weights `w_ij` | Fast (per episode) |
| **Outer loop (evolution)** | Genetic algorithm | Plasticity parameters `{A, B, C, D, η}` and possibly architecture | Slow (across generations) |
| **Policy behavior** | Emergent | Mapping `s_t → a_t` that maximizes reward | Indirectly learned |

---

## 🧩 ABCD Hebbian Plasticity Rule

At each timestep `t`, synaptic weights are updated using:

\[
\Delta w_{ij}(t) = η \, r_t \, (A \, x_i y_j + B \, x_i + C \, y_j + D)
\]

Optionally, with eligibility traces:

\[
e_{ij}(t) = λ e_{ij}(t-1) + (A x_i y_j + B x_i + C y_j + D)
\]
\[
\Delta w_{ij}(t) = η \, r_t \, e_{ij}(t)
\]

where:

- `x_i`: presynaptic activation  
- `y_j`: postsynaptic activation  
- `r_t`: reward or modulatory signal  
- `η`: learning rate  
- `A, B, C, D`: coefficients controlling each plasticity term  

---

## 🧭 Full Training Pipeline

### **1. Population Initialization**

- Initialize a population of `N` agents.  
- Each agent has:
  - Random synaptic weights `w_ij`
  - A set of **plasticity parameters** `{A, B, C, D, η}`
  - (Optional) structural parameters like layer sizes or connection probabilities

---

### **2. Inner Loop — Lifetime Learning (Hebbian Updates)**

For each agent:

1. **Reset** the environment and initialize neural activations.
2. **Repeat for each timestep `t` in an episode:**
   - Observe state `s_t`.
   - Compute neuron activations:  
     `y_j = f(Σ_i w_ij x_i)`
   - Sample an action:  
     `a_t ∼ π(s_t)`
   - Execute `a_t`, observe reward `r_t` and next state `s_{t+1}`.
   - Apply the **Hebbian ABCD update** for each synapse:
     ```
     Δw_ij = η * r_t * (A*x_i*y_j + B*x_i + C*y_j + D)
     w_ij ← w_ij + Δw_ij
     ```
   - Optionally clip or normalize weights.

3. Continue until episode ends or time limit is reached.

---

### **3. Fitness Evaluation**

After each agent’s episode(s):

\[
F_i = \sum_t r_t
\]

This fitness score measures how well the evolved plasticity parameters allowed the agent to learn effective behavior during its lifetime.

---

### **4. Outer Loop — Evolutionary Optimization**

1. **Selection:** Choose top-performing individuals by fitness.  
2. **Reproduction:** Copy their plasticity parameters `{A, B, C, D, η}`.  
3. **Mutation:** Apply Gaussian noise or perturbation to each parameter.  
4. **Crossover (optional):** Combine parameters between parents.  
5. **Next Generation:** Replace the old population with offspring.

Repeat this process over many generations until convergence or performance plateau.

---

## 🧩 Optional: Hybrid Policy Setup

A **policy head** can be placed on top of the Hebbian substrate:

- The Hebbian layers learn internal feature representations via local plasticity.
- The policy head converts those representations into actions, trained by gradient-based or evolutionary updates.

This hybrid combines **fast local adaptation** with **efficient global control**.

---

## 🌱 Outcome

After multiple generations:
- The EA evolves **effective plasticity parameters** that allow agents to self-adapt.
- The Hebbian mechanism enables **online learning** without backpropagation.
- The policy exhibits emergent intelligent behavior optimized through evolution.

This system bridges **biologically plausible learning** with **reinforcement optimization**, evolving *not just weights*, but the **rules of learning themselves**.

--- -->