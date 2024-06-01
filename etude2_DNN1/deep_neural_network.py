#!/usr/bin/env python
# coding: utf-8

# **Steps**:
# - Same as a shallow neural network but with more hidden layer
# - Relu activation function for hidden layer
# - Sigmoid activation function for final layer (1 neuron)

# **Notation**:
# - Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer. 
#     - Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
# - Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example. 
#     - Example: $x^{(i)}$ is the $i^{th}$ training example.
# - Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
#     - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
# 

# In[1]:


import numpy as np
import h5py
import matplotlib.pyplot as plt

import copy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


def sigmoid(Z):
    """
    Compute the sigmoid of Z

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu(Z):
    """
    Implement the ReLU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dA to a correct object.
    
    # When Z <= 0, set dZ to 0.
    dZ[Z <= 0] = 0
    return dZ


# <a name='2'></a>
# **Outline:**
# 
# To build your neural network, you'll be implementing several "helper functions." These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. 
# 
# Each small helper function will have detailed instructions to walk you through the necessary steps. Here's an outline of the steps:
# 
# - Initialize the parameters for a two-layer network and for an $L$-layer neural network
# - Implement the forward propagation module (shown in purple in the figure below)
#      - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
#      - The ACTIVATION function is provided for you (relu/sigmoid)
#      - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#      - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
# - Compute the loss
# - Implement the backward propagation module (denoted in red in the figure below)
#     - Complete the LINEAR part of a layer's backward propagation step
#     - The gradient of the ACTIVATION function is provided for you(relu_backward/sigmoid_backward) 
#     - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
#     - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# - Finally, update the parameters
# 
# <img src="images/final outline.png" style="width:800px;height:500px;">
# <caption><center><b>Figure 1</b></center></caption><br>
# 
# 
# **Note**:
# 
# For every forward function, there is a corresponding backward function. This is why at every step of your forward module you will be storing some values in a cache. These cached values are useful for computing gradients. 
# 
# In the backpropagation module, you can then use the cache to calculate the gradients.

# ## 1 - Initialization for a 2 layer Neural Network
# 
# Write two helper functions to initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one generalizes this initialization process to $L$ layers.
# 
# **Instructions**:
# - The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 

# In[3]:


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))
    # YOUR CODE ENDS HERE
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# ## 1$_{bis}$ - Initialization for a L layer Neural Network
# 
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep` function, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. For example, if the size of your input $X$ is $(12288, 209)$ (with $m=209$ examples) then:
# 
# <table style="width:100%">
#     <tr>
#         <td>  </td> 
#         <td> <b>Shape of W</b> </td> 
#         <td> <b>Shape of b</b>  </td> 
#         <td> <b>Activation</b> </td>
#         <td> <b>Shape of Activation</b> </td> 
#     <tr>
#     <tr>
#         <td> <b>Layer 1</b> </td> 
#         <td> $(n^{[1]},12288)$ </td> 
#         <td> $(n^{[1]},1)$ </td> 
#         <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td> 
#         <td> $(n^{[1]},209)$ </td> 
#     <tr>
#     <tr>
#         <td> <b>Layer 2</b> </td> 
#         <td> $(n^{[2]}, n^{[1]})$  </td> 
#         <td> $(n^{[2]},1)$ </td> 
#         <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td> 
#         <td> $(n^{[2]}, 209)$ </td> 
#     <tr>
#        <tr>
#         <td> $\vdots$ </td> 
#         <td> $\vdots$  </td> 
#         <td> $\vdots$  </td> 
#         <td> $\vdots$</td> 
#         <td> $\vdots$  </td> 
#     <tr>  
#    <tr>
#        <td> <b>Layer L-1</b> </td> 
#         <td> $(n^{[L-1]}, n^{[L-2]})$ </td> 
#         <td> $(n^{[L-1]}, 1)$  </td> 
#         <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td> 
#         <td> $(n^{[L-1]}, 209)$ </td> 
#    <tr>
#    <tr>
#        <td> <b>Layer L</b> </td> 
#         <td> $(n^{[L]}, n^{[L-1]})$ </td> 
#         <td> $(n^{[L]}, 1)$ </td>
#         <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
#         <td> $(n^{[L]}, 209)$  </td> 
#     <tr>
# </table>
# 
# Remember that when you compute $W X + b$ in python, it carries out broadcasting. For example, if: 
# 
# $$ W = \begin{bmatrix}
#     w_{00}  & w_{01} & w_{02} \\
#     w_{10}  & w_{11} & w_{12} \\
#     w_{20}  & w_{21} & w_{22} 
# \end{bmatrix}\;\;\; X = \begin{bmatrix}
#     x_{00}  & x_{01} & x_{02} \\
#     x_{10}  & x_{11} & x_{12} \\
#     x_{20}  & x_{21} & x_{22} 
# \end{bmatrix} \;\;\; b =\begin{bmatrix}
#     b_0  \\
#     b_1  \\
#     b_2
# \end{bmatrix}\tag{2}$$
# 
# Then $WX + b$ will be:
# 
# $$ WX + b = \begin{bmatrix}
#     (w_{00}x_{00} + w_{01}x_{10} + w_{02}x_{20}) + b_0 & (w_{00}x_{01} + w_{01}x_{11} + w_{02}x_{21}) + b_0 & \cdots \\
#     (w_{10}x_{00} + w_{11}x_{10} + w_{12}x_{20}) + b_1 & (w_{10}x_{01} + w_{11}x_{11} + w_{12}x_{21}) + b_1 & \cdots \\
#     (w_{20}x_{00} + w_{21}x_{10} + w_{22}x_{20}) + b_2 &  (w_{20}x_{01} + w_{21}x_{11} + w_{22}x_{21}) + b_2 & \cdots
# \end{bmatrix}\tag{3}  $$
# 

# **Instructions**:
# - The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# 

# In[4]:


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters


# ## 2 - Forward propagation module
# 
# Now that you have initialized your parameters, you can do the forward propagation module. Start by implementing some basic functions that you can use again later when implementing the model. Now, you'll complete three functions in this order:
# 
# - LINEAR
# - LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
# - [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID (whole model)
# 
# The linear forward module (vectorized over all the examples) computes the following equations:
# 
# $$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\tag{4}$$
# 
# where $A^{[0]} = X$. 

# In[5]:


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A) + b

    cache = (A, W, b)
    
    return Z, cache


# In this notebook, you will use two activation functions:
# 
# - **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. You've been provided with the `sigmoid` function which returns **two** items: the activation value "`a`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call: 
# ``` python
# A, activation_cache = sigmoid(Z)
# ```
# 
# - **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. You've been provided with the `relu` function. This function returns **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`" (it's what you'll feed in to the corresponding backward function). To use it you could just call:
# ``` python
# A, activation_cache = relu(Z)
# ```
# 
# For added convenience, you're going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION). Hence, you'll implement a function that does the LINEAR forward step, followed by an ACTIVATION forward step.

# In[6]:


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache


# **L-Layer Model**
# 
# For even *more* convenience when implementing the $L$-layer Neural Net, you will need a function that replicates the previous one (`linear_activation_forward` with RELU) $L-1$ times, then follows that with one `linear_activation_forward` with SIGMOID.
# 
# <img src="images/model_architecture_kiank.png" style="width:600px;height:300px;">
# <caption><center> <b>Figure 2</b> : *[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model</center></caption><br>
# 
# Implement the forward propagation of the above model.
# 
# **Instructions**: 
# - Implement the forward propagation of the above model.
# - In the code below, the variable `AL` will denote $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is sometimes also called `Yhat`, i.e., this is $\hat{Y}$.) 
# 
# **Hints**:
# - Use the functions you've previously written 
# - Use a for loop to replicate [LINEAR->RELU] (L-1) times
# - Don't forget to keep track of the caches in the "caches" list.

# In[7]:


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b" + str(l)], 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b" + str(L)], 'sigmoid')    
    caches.append(cache)
    
    return AL, caches


# ## 3 - Cost function
# Now you can implement forward and backward propagation! You need to compute the cost, in order to check whether your model is actually learning.
# 
# Compute the cross-entropy cost $J$, using the following formula: $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$

# In[8]:


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    cost = -(1/m) * ( np.dot(Y,np.log(AL.T)) + np.dot(1-Y,np.log(1-AL.T)))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


# ## 4 - Backward propagation module
# 
# 
# Just as you did for the forward propagation, you'll implement helper functions for backpropagation. Remember that backpropagation is used to calculate the gradient of the loss function with respect to the parameters. 
# 
# **Reminder**: 
# <img src="images/backprop_kiank.png" style="width:650px;height:250px;">
# <caption><center><font color='purple'><b>Figure 3</b>: Forward and Backward propagation for LINEAR->RELU->LINEAR->SIGMOID <br> <i>The purple blocks represent the forward propagation, and the red blocks represent the backward propagation.</font></center></caption>
# 
# 
# <!-- 
# For those of you who are experts in calculus (which you don't need to be to do this assignment!), the chain rule of calculus can be used to derive the derivative of the loss $\mathcal{L}$ with respect to $z^{[1]}$ in a 2-layer network as follows:
# 
# $$\frac{d \mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \frac{d\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\frac{{da^{[2]}}}{{dz^{[2]}}}\frac{{dz^{[2]}}}{{da^{[1]}}}\frac{{da^{[1]}}}{{dz^{[1]}}} \tag{8} $$
# 
# In order to calculate the gradient $dW^{[1]} = \frac{\partial L}{\partial W^{[1]}}$, use the previous chain rule and you do $dW^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial W^{[1]}}$. During backpropagation, at each step you multiply your current gradient by the gradient corresponding to the specific layer to get the gradient you wanted.
# 
# Equivalently, in order to calculate the gradient $db^{[1]} = \frac{\partial L}{\partial b^{[1]}}$, you use the previous chain rule and you do $db^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial b^{[1]}}$.
# 
# This is why we talk about **backpropagation**.
# !-->
# 
# Now, similarly to forward propagation, you're going to build the backward propagation in three steps:
# 1. LINEAR backward
# 2. LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
# 3. [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)

# For the next exercise, you will need to remember that:
# 
# - `b` is a matrix(np.ndarray) with 1 column and n rows, i.e: b = [[1.0], [2.0]] (remember that `b` is a constant)
# - np.sum performs a sum over the elements of a ndarray
# - axis=1 or axis=0 specify if the sum is carried out by rows or by columns respectively
# - keepdims specifies if the original dimensions of the matrix must be kept.
# - Look at the following example to clarify:

# In[9]:


A = np.array([[1, 2], [3, 4]])

print('axis=1 and keepdims=True')
print(np.sum(A, axis=1, keepdims=True))
print('axis=1 and keepdims=False')
print(np.sum(A, axis=1, keepdims=False))
print('axis=0 and keepdims=True')
print(np.sum(A, axis=0, keepdims=True))
print('axis=0 and keepdims=False')
print(np.sum(A, axis=0, keepdims=False))


# **Linear Backward (1 itaration)**
# 
# For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).
# 
# Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$.
# 
# <img src="images/linearback_kiank.png" style="width:250px;height:300px;">
# <caption><center><font color='purple'><b>Figure 4</b></font></center></caption>
# 
# The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$ are computed using the input $dZ^{[l]}$.
# 
# Here are the formulas you need:
# $$ dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$
# $$ db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$
# $$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$
# 
# 
# $A^{[l-1] T}$ is the transpose of $A^{[l-1]}$. 

# In[10]:


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims= True)/m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# **Linear-Activation Backward (1 iteration)**
# 
# Next, you will create a function that merges the two helper functions: **`linear_backward`** and the backward step for the activation **`linear_activation_backward`**. 
# 
# To help you implement `linear_activation_backward`, two backward functions have been provided:
# - **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit. You can call it as follows:
# 
# ```python
# dZ = sigmoid_backward(dA, activation_cache)
# ```
# 
# - **`relu_backward`**: Implements the backward propagation for RELU unit. You can call it as follows:
# 
# ```python
# dZ = relu_backward(dA, activation_cache)
# ```
# 
# If $g(.)$ is the activation function, 
# `sigmoid_backward` and `relu_backward` compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}). \tag{11}$$  
# 

# In[11]:


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# **Implementation for a L-layer model (L iterations)**
# 
# Now you will implement the backward function for the whole network! 
# 
# Recall that when you implemented the `L_model_forward` function, at each iteration, you stored a cache which contains (X,W,b, and z). In the back propagation module, you'll use those variables to compute the gradients. Therefore, in the `L_model_backward` function, you'll iterate through all the hidden layers backward, starting from layer $L$. On each step, you will use the cached values for layer $l$ to backpropagate through layer $l$. Figure 5 below shows the backward pass. 
# 
# 
# <img src="images/mn_backward.png" style="width:450px;height:300px;">
# <caption><center><font color='purple'><b>Figure 5</b>: Backward pass</font></center></caption>
# 
# **Initializing backpropagation**:
# 
# To backpropagate through this network, you know that the output is: 
# $A^{[L]} = \sigma(Z^{[L]})$. Your code thus needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
# To do so, use this formula (derived using calculus which, again, you don't need in-depth knowledge of!):
# ```python
# dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
# ```
# 
# You can then use this post-activation gradient `dAL` to keep going backward. As seen in Figure 5, you can now feed in `dAL` into the LINEAR->SIGMOID backward function you implemented (which will use the cached values stored by the L_model_forward function). 
# 
# After that, you will have to use a `for` loop to iterate through all the other layers using the LINEAR->RELU backward function. You should store each dA, dW, and db in the grads dictionary. To do so, use this formula : 
# 
# $$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$
# 
# For example, for $l=3$ this would store $dW^{[l]}$ in `grads["dW3"]`.
# 
# **Instruction:**
# 
# - Implement backpropagation for the *[LINEAR->RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model.

# In[12]:


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads['dA' + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# ## 5 - Uptade parameters
# 
# In this section, you'll update the parameters of the model, using gradient descent: 
# 
# $$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
# $$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$
# 
# where $\alpha$ is the learning rate. 
# 
# After computing the updated parameters, store them in the parameters dictionary. 
# 
# **Instructions**:
# 
# - Update parameters using gradient descent on every $W^{[l]}$ and $b^{[l]}$ for $l = 1, 2, ..., L$. 

# In[13]:


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


# ## 6 - Full model (2-layer neural network)
# 
# **Instructions:**
# 
# - Define a function to make predictions
# - Use the helper functions you have implemented in the previous assignment to build a 2-layer neural network with the following structure: *LINEAR -> RELU -> LINEAR -> SIGMOID*. The functions you may need and their inputs are:
# ```python
# def initialize_parameters(n_x, n_h, n_y):
#     ...
#     return parameters 
# def linear_activation_forward(A_prev, W, b, activation):
#     ...
#     return A, cache
# def compute_cost(AL, Y):
#     ...
#     return cost
# def linear_activation_backward(dA, cache, activation):
#     ...
#     return dA_prev, dW, db
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
# ```

# In[14]:


# definition of the predict function, to make our predictions

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


# In[15]:


### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


# In[16]:


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
   
    parameters = initialize_parameters(n_x, n_h, n_y)  # Initialize parameters dictionary

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")    
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid") 

        # Compute cost
        cost = compute_cost(A2, Y) 

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2)) 

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        # Setting gradients to update parameters
        grads['dW1'] = dW1 
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate) 

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# **Implementation**

# In[17]:


#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


# In[18]:


#predictions_train = predict(train_x, train_y, parameters)


# In[19]:


#predictions_test = predict(test_x, test_y, parameters)


# ## 6$_{bis}$ - Full model (L-layer neural network)
# 
# **Instructions:**
# 
# - Use the helper functions you have implemented previously to build an $L$-layer neural network with the following structure: *[LINEAR -> RELU]$\times$(L-1) -> LINEAR -> SIGMOID*. The functions you may need and their inputs are:
# ```python
# def initialize_parameters_deep(layers_dims):
#     ...
#     return parameters 
# def L_model_forward(X, parameters):
#     ...
#     return AL, caches
# def compute_cost(AL, Y):
#     ...
#     return cost
# def L_model_backward(AL, Y, caches):
#     ...
#     return grads
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
# ```

# In[20]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# In[21]:


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[22]:


#parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[23]:


#pred_train = predict(train_x, train_y, parameters)


# In[24]:


#pred_test = predict(test_x, test_y, parameters)


# This is how we implement a deep neural network.
# 
# This model will serve as a benchmark model for the finance project linked to this repository. 
# 
# To make accurate predictive asumption we will see how to finetune hyperparameters, use regularisation techniques and implement gradient checking.
