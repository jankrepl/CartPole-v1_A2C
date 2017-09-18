# CartPole_v1 A2C
Solution to the CartPole_v1 environment using the Advantage Actor Critic (**A2C**) algorithm

## Code

### Running
```
python Main.py
```

### Dependencies
*  gym
*  numpy
*  tensorflow

## Detailed Description
### Problem Statement and Environment
The environment is identical to CartPole-v0 but the number of required timesteps is increased to 475.

---
### A2C Algorithm
We need to approximate two separate functions:
* Actor's policy
* Critic's state value function

Both of them are modeled by a neural network with one hidden layer. 

#### Actor - policy
Policy is a mapping from the state space to a set of probability distributions over action space (in our case only discrete).
To each given state we assign a 2-element probability vector whose elements sum up to 1.


#### Critic - state value function
State value function is a mapping from the state space to the real numbers. To each given state we assign a real number
representing a "value/utility" of being at that state.

#### Training
For *actor* we are directly optimizing in the policy space. To do that we use the Policy Gradient Theorem and a below
TD(0) estimate of the **advantage function**

![screen shot 2017-09-18 at 9 19 12 am](https://user-images.githubusercontent.com/18519371/30531986-088161ec-9c52-11e7-925c-283d7f9abb5a.png)

This can be reformulated in terms of a **cross entropy loss minimization**.


The **critic** is always updated based on the TD(0) back up. This can be reformulated in terms of a **squared error loss minimization**.
## Results and discussion
This method seems to converge no matter what the initialization is. See below an evolution of scores for one run.
![screen shot 2017-09-18 at 8 31 17 am](https://user-images.githubusercontent.com/18519371/30531566-7ca98610-9c4f-11e7-95ae-837e57dd7221.png)


## Resources and links
* ![RLCode](https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/4-actor-critic) - Similar algorithm in Keras and same hyperparameters
* ![David Silver - Policy Gradient Lecture Slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
