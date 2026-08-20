# Under construction

Come back later!

## Heading
### Heading
#### Heading


$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))
$$

:::note
VAE uses a latent distribution to encode inputs.
:::

:::tip
Always normalize your inputs for better performance.
:::

:::warning
Do not forget to tune the KL term weight.
:::

:::idea
You can combine the encoder and decoder losses for a hybrid model.
:::

:::example
```python
import numpy as np

print("Hello World!")
```
:::

:::theory
The KL divergence term ensures the latent distribution remains close to a prior, typically $\mathcal{N}(0, I)$.
:::

### link ex
- [VQVAE](page.html?topic=vqvae)
- [PPO](page.html?topic=ppo)

![Sample image](imgs/robot.png "Sample imgs")