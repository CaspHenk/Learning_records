# Quantum Basics

## Introduction

This whole section comes from my own interest in the field of quantum computing. I started by following IBM courses, and these pages are a summary of what I learned there, in my own words. 


## Notation

### "Bra" and "Ket"

These are the words to define a certain type of vectors. $\bra{0} = \begin{pmatrix} 1 & 0 \end{pmatrix}$ reads "Bra 0", and $\ket{0} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ reads "Ket 0". These are very useful in many cases. For example let's say you have a quantum state vector (see below for the definition) $\ket{\psi} = \sum_{i=0}^N \alpha_i \ket{i}$, where $\alpha_i$ are the entries of the quantum state vector, then $\bra{i}\ket{\psi} = \braket{i | \psi} = \alpha_i$. This notation is used a lot when computing measurement probabilities, e.g. "What's the probability of measuring $i$ at the output of this system?" You'd just have to compute $\alpha_i^2$ from the output quantum state vector, and to put it nicely in maths you use this property.

## Basic concepts

First, I will compare known concepts of classical information theory with quantum information, to make it easier to see where the difference is:

- **System**: a place where information (states) is stored.

- **Classical state**: the state of the information in a system. For example, a die with 6 faces has 6 different classical states, and the die itself could be considered to be the system.

- **Probability vector**: we can assign probabilities to each state of a system, which then gives a probability vector. This vector obviously contains only *non-negative real numbers* and *the sum of its elements is equal to 1*. Examples: 

$$
\begin{pmatrix} 1/2 \\ 1/4 \\ 1/4 \end{pmatrix} \, , \, \begin{pmatrix} 1/3 \\ 2/3 \end{pmatrix}
$$

- **Stochastic matrix**: matrices whose *columns* form probability vectors. Examples: 

$$
\begin{pmatrix} 1/2 & 1/3 & 1 \\
1/4 & 1/3 & 0 \\
1/4 & 1/3 & 0\end{pmatrix} \, , \,
\begin{pmatrix} 3/4 & 1/3 \\
1/4 & 2/3\end{pmatrix}
$$

- **Quantum state vector**: represented by a column vector, and its entries represent the classical states of the system. However, the main difference with a classical state vector is that here the entries are **complex numbers** and the **sum of absolute values squared of the entries is 1** (implies that quantum state vectors are *unit* vectors).

- **Qubit**: it's a quantum system whose classical state set is ${0,1}$. Basically, it's a bit that can be in a quantum state, which means that its description determines the probabilities of its future behaviour.


