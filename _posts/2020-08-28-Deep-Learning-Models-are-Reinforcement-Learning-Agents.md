---  
layout: post  
title:  "Deep Learning Models are Reinforcement Learning Agents"  
author: pawan  
categories: [Deep Learning, Reinforcement Learning ]  
image: assets/images/post_id_20200828/RLDLBlock.svg  
featured: true  
---  

Deep learning has reached its heights while Reinforcement learning is yet to find its moment. But today we will take a problem from the famous deep learning space and map it to not so famous RL space. We would explore an alternative version of training the deep learning models and explore how two are trying to solve the same problem.

## Formulating  a Deep Learning Model Training as RL Problem
We often argue that training a deep learning problem is easier as the reward is immediate while in RL space the reward is after multiple steps or delayed reward. This looks true at the problem definition stage, however, training a neural network has multiple stages and loss is only calculated at the terminal layer and then back-propagated to earlier layers, this finds stark similarity to the RL reward discounting, which is a deterministic way to pass back the loss.

Let us take a deep learning layer from the model and represent it as an RL agent.

![svg]({{site.baseurl}}/assets/images/post_id_20200828/RLDLBlock.svg)

Now each block has a state input $s_t$ from preceding layer and takes an action $s_t$ and maps the states to new state $s_{t+1}$. Each Block is markovian as it depends only on the preceding input and not the history. It is a case of Model-Based Reinforcement Learning, but for deep learning blocks, we can conveniently assume the ENV to be the Identity function.


To formulate the deep learning training as an RL problem we can draw parallel as per this diagram.

![svg]({{site.baseurl}}/assets/images/post_id_20200828/RLDLDiagram.svg)

Where

- $s_0$ is the input vector $X$ to the neural network.
- Each block can be considered as a policy function which maps the previous state $s_0$ to $s_1$
- Each block since depends only on the immediate input is a markovian block.
- Finally, the output of the final layer is compared with output $Y$ and the negative of the loss function can be used as a reward function.
- We want to maximise reward or minimise the loss.
- Each Policy function(block) is parameterised by $\theta$.
- Goal of this RL agent is find the state, action sequence(state-space trajectory) $s_0, a_0, s_1, a_1, ... , a_{n-1}, s_n $ that maximises the rewards(or minimises the loss)

### Multiple Policy Functions

In the above problem, we see each deep learning block is an independent policy function. Can we use a single policy function to make the problem look more like an academic RL problem?

In deep learning space, we already have a concept of parameter sharing, for example, ALBERT is a parameter shared version of the BERT. While BERT will look more like the problem described above. ALBERT, since replicate the layer n times, is RL corollary of our problem with single policy function used in each action.

### Variable State/Path Length

While in RL we generally have variable path length depending on the initial state. In deep learning corollary, we will often have a fixed number of state transitions(= number of layers). 

However [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) Paper proposes a method of variable path length neural networks. Where you can apply the same network layer multiple times and proceed further on the state space trajectory. 

This also sets the stage to use RL as the Neural ~~Architecture~~ Depth Search(NAS) problem for weight shared deep learning.

### Pros for Deep Learning
- Training Deep Learning Architecture is hard due to loss degradations. (A problem [ResNets](https://arxiv.org/abs/1512.03385) have tackled some years back). This using the learnings from Reinforcement Learning can pass loss(reward) from the last layer to early layers solve with it.
- We can now use non-differential blocks as policy blocks and optimise networks.

### Pros for the Reinforcement Learning
Reinforcement learning can formulate the RL problems as weight shared deep neural networks and pass on the losses by using differential layers.

We can formulate the RL problem as a deep learning problem now. The maze transversal RL problem is essentially a Variable Length-Weight Shared neural network policy function block.

### Code Example
Now let us see how the state space trajectory of a neural network looks with a code example.

We would take a 2 dimensional(for plotting ease) 2 layer output layer with multi-depth(so the trajectory is meaningful) to observe the trajectory

```python
# Neural Network
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Flatten, Dot, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy

# Model
model = Sequential()
model.add(Dense(2, input_shape=(n_dim,), activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss= SparseCategoricalCrossentropy(), 
              optimizer=tf.keras.optimizers.Adam(),
              metrics = "accuracy")
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 6         
    =================================================================
    Total params: 24
    Trainable params: 24
    Non-trainable params: 0
    _________________________________________________________________



```python
# Train Model
model.fit(X, y, batch_size=1, epochs=20, validation_split=0.2)
```

    Epoch 20/20
    800/800 [==============================] - 1s 2ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000



    <tensorflow.python.keras.callbacks.History at 0x7fb729bb9e10>




```python
# Extract Model Layers
model_layers = model.layers

# Plot 10 Trajectories
plot_n = 10 
layer_input = X[0:plot_n]
trajectories = [layer_input,]
for alayer in model_layers:
    layer_output = alayer(layer_input)
    trajectories.append(layer_output)
    layer_input = layer_output


## Arrow Plots
from matplotlib.patches import FancyArrowPatch 
import matplotlib.pyplot as plt


def arrow(x,y,ax,n):
    d = len(x)//(n+1)    
    ind = np.arange(d,len(x),d)
    for i in ind:
        ar = FancyArrowPatch ((x[i-1],y[i-1]),(x[i],y[i]), arrowstyle='->', mutation_scale=20)
        ax.add_patch(ar)


## Plot
fig, ax = plt.subplots()
ax.set_xticks(np.arange(-1.1, 1.1, 0.1))
ax.set_yticks(np.arange(-1.1, 1.1, 0.1))

for idx in range(plot_n):    
    # line
    x_tracjectory = [tensor_var[idx, 0] for tensor_var in trajectories]
    y_tracjectory = [tensor_var[idx, 1] for tensor_var in trajectories]
    # plotting the line color as per class.
    color = 'b' if y[idx]==0 else 'r'
    ax.plot(x_tracjectory, y_tracjectory, c=color)
    arrow(x_tracjectory, y_tracjectory, ax, 2)

# Graph Annotation
plt.title('State Trajectory Comparision of Two Classes')
plt.grid(True)
plt.show()
```
#### Five(Input + 4 layer) Step Trajectory Comparision of Two Classes.

![SVG]({{site.baseurl}}/assets/images/post_id_20200828/StateTrajectories.svg)

This is only indicative as neural networks have multiple solutions to the same dataset we can see different trajectories.
Also since we have increased the depth just for plotting, it is not the optimal trajectory path. we can see neural network wobbles around in the space.

A solution to the above toy problem, we will leave to a future post.