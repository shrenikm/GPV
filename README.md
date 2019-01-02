# GPV - Generation of Policy Variations in Deep Reinforcement Learning

Reinforcement learning agents usually learn a deterministic policy which is fixed for a particular input state. In cases where the environment of the agent is different from when the episodic data was collected, it is advantageous to be able to vary this policy, while still keeping it consistent with the goal of solving the given task.

GPV uses generative models to generate variations in the policy learned by an Agent. In this implementation, conditional VAEs are used to generate the policy. The VAE maximizes the probability of the output action, given the state and noise from a multivariate normal distribution. It ensures that instead of computing a single action for any given state, it computes samples from a state conditioned distribution around the optimal policy.

## Requirements and execution

The algorithm was tested using:
- Python 3.6.7
- PyTorch 1.0

The files that need to be executed are the `run` files in the parent directory.

## Post-Training GPV

In this implementation, an RL agent is trained using Deep Deterministic Policy Gradients (DDPG). A VAE is trained based on the output of the DDPG actor, which is then used to generate varying policies, given the current state of the system.

### Results

The OpenAI gym bipedal environment was used to evaluate the results of the algorithm. 

The weights of the reconstruction and KL Divergence Loss in the VAE can be tuned to change the behavior of the generated policy. If the reconstruction loss is weighted more, it leads to actions that are more faithful to the DDPG policy. If the KL Divergence loss is weighted more, it leads to more variation in the actions.

The gif below shows the biped running the DDPG policy. The gifs below that show the biped using the generated policies, both from the same VAE.

<p align="center">
<img src="https://media.giphy.com/media/ddx0IwLxYIQfyAb2MW/giphy.gif" width="420" />
</p>

<p align="center">
<img src="https://media.giphy.com/media/bEUx2P6oYc0rrjG17E/giphy.gif" width="420" />
<img src="https://media.giphy.com/media/Zw4tEpQ3Hcxx0xX3XC/giphy.gif" width="420" /> 
</p>
