# Hebbian learning

Hebbian learning is based on the use of **Hebbian controllers** a special type of controller that does not use backpropagation for weight updates but bases itself on the so-called ABCD rules, which are correlation-based weights:

:::theory

$$\Delta w_{ij} = A_{ij} i_i o_j + B_{ij} i_i + C_{ij} o_j + D_{ij}
$$

:::


:::question

#### Why does a Hebbian controller need a Policy to be able to learn something?

:::


:::question

#### 