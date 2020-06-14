---
layout: post
title:  "Multi Single Class Learning"
author: pawan
categories: [ Loss Function, Deep Learning ]
image: assets/images/post_id_20200612/output_4_1.png
featured: true
---

Machine Learning requires large amount of clean data for the models to be trained. But thats rarely the case in reality. A Scenario in real life is clicks/likes data, where a click signifies the customer preference. But this is only positive class and for machine learning we need negative classes too.

This seems like OneVsAll problem at the face of it but its not in the real sense. Because "VsAll" assumes that the sample doesn't belong to other classes. But in our problem statement we may not so clearly seprable classes. So overlapping decision boundaries.

Lets illustrate it with a data example.


```python
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
```


```python
# Lets generate 1000 samples in 3 classes in 2 dimensional space
n_classes = 3
n_dim = 2
n_samples = 1000
X, y_original = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_dim, random_state=100)
print(X.shape)

# Plot Data
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_original, edgecolor='k')
plt.title("Plot for Sample")
plt.show()
```

    (1000, 2)



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_4_1.png)


This looks like very desirable data with 3 classes really seperable. But we seldom gets data so clean in real life. So lets introduce some confusion with randomly merging class 2 into 0 and 1.


```python
import numpy as np
print("Distinct Values in y(Before):", np.unique(y_original))
y = np.copy(y_original)
for i in range(y_original.shape[0]):
  if y[i] == n_classes-1:
    y[i] = np.random.choice(2, 1)
print("Distinct Values in y(After):", np.unique(y))

# Plot Data
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolor='k')
plt.title("Plot for Sample")
plt.show()
```

    Distinct Values in y(Before): [0 1 2]
    Distinct Values in y(After): [0 1]



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_6_1.png)


This looks like quite a real life scenario. Where the tag says it belongs to this class doesn't mean its not likely to be part of other class. Or often the positive class tags are only collected. Let classify these with deep learning.

## Classification Model

### Softmax Classification


```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Flatten, Dot, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy
```


```python
# Model
model = Sequential()
model.add(Dense(n_classes, input_shape=(n_dim,), activation='relu'))
model.add(Dense(n_classes-1, activation='softmax'))
model.compile(loss= SparseCategoricalCrossentropy(), 
              optimizer=tf.keras.optimizers.Adam(),
              metrics = "accuracy")
model.summary()

# Train Model
model.fit(X, y, batch_size=1, epochs=10, validation_split=0.2)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 3)                 9         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 8         
    =================================================================
    Total params: 17
    Trainable params: 17
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.5257 - accuracy: 0.7125 - val_loss: 0.3325 - val_accuracy: 0.8450
    Epoch 2/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.3122 - accuracy: 0.8138 - val_loss: 0.2714 - val_accuracy: 0.8300
    Epoch 3/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2821 - accuracy: 0.8263 - val_loss: 0.2739 - val_accuracy: 0.8300
    Epoch 4/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2844 - accuracy: 0.8163 - val_loss: 0.2588 - val_accuracy: 0.8300
    Epoch 5/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2778 - accuracy: 0.8325 - val_loss: 0.2588 - val_accuracy: 0.8350
    Epoch 6/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2743 - accuracy: 0.8213 - val_loss: 0.2490 - val_accuracy: 0.8500
    Epoch 7/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2727 - accuracy: 0.8225 - val_loss: 0.2507 - val_accuracy: 0.8450
    Epoch 8/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2715 - accuracy: 0.8200 - val_loss: 0.2489 - val_accuracy: 0.8450
    Epoch 9/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2697 - accuracy: 0.8200 - val_loss: 0.2492 - val_accuracy: 0.8350
    Epoch 10/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2691 - accuracy: 0.8225 - val_loss: 0.2534 - val_accuracy: 0.8350





    <tensorflow.python.keras.callbacks.History at 0x7f6a0c97a278>




```python
# Plot Data
y_predicted = model.predict_classes(X)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_predicted, edgecolor='k')
plt.title("Plot for Predicted Class Softmax")
plt.show()
```

    WARNING:tensorflow:From <ipython-input-6-e4a73b563ee1>:2: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_12_1.png)



```python
y_score = model.predict(X)
for i in range(n_classes):
  idx = np.where(y_original==i)
  mean_score = [y_score[idx,j].mean() for j in range(n_classes-1)]
  print("Class Idx: {0}".format(i), mean_score)
```

    Class Idx: 0 [0.97182953, 0.028170489]
    Class Idx: 1 [0.00021615, 0.9997839]
    Class Idx: 2 [0.59249353, 0.40750644]


This clearly has partitioned the class into two. Which is not the intended case, Also it can't be expected from CrossEntropy as it clearly a multi class paritioning loss. Lets try with BinaryCrossEntropy

### BinaryCrossEntopy


```python
# Model
model = Sequential()
model.add(Dense(n_classes-1, input_shape=(n_dim,), activation='relu'))
model.add(Dense(n_classes-1, activation='sigmoid'))
model.compile(loss= BinaryCrossentropy(), 
              optimizer=tf.keras.optimizers.Adam(),
              metrics = "accuracy")
model.summary()

# Train Model
y_train = tf.keras.utils.to_categorical(y, num_classes=n_classes-1)
model.fit(X, y_train, batch_size=1, epochs=10, validation_split=0.2)
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_2 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 6         
    =================================================================
    Total params: 12
    Trainable params: 12
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.7239 - accuracy: 0.4538 - val_loss: 0.6611 - val_accuracy: 0.7250
    Epoch 2/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.5078 - accuracy: 0.8225 - val_loss: 0.3412 - val_accuracy: 0.8300
    Epoch 3/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2909 - accuracy: 0.8313 - val_loss: 0.2480 - val_accuracy: 0.8300
    Epoch 4/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2495 - accuracy: 0.8313 - val_loss: 0.2308 - val_accuracy: 0.8300
    Epoch 5/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2407 - accuracy: 0.8313 - val_loss: 0.2255 - val_accuracy: 0.8300
    Epoch 6/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2377 - accuracy: 0.8313 - val_loss: 0.2231 - val_accuracy: 0.8300
    Epoch 7/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2361 - accuracy: 0.8313 - val_loss: 0.2218 - val_accuracy: 0.8300
    Epoch 8/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2354 - accuracy: 0.8313 - val_loss: 0.2208 - val_accuracy: 0.8300
    Epoch 9/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2346 - accuracy: 0.8313 - val_loss: 0.2203 - val_accuracy: 0.8300
    Epoch 10/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.2345 - accuracy: 0.8313 - val_loss: 0.2199 - val_accuracy: 0.8300





    <tensorflow.python.keras.callbacks.History at 0x7f6a09ab67b8>




```python
# Plot Data
y_predicted = model.predict(X)

# Now we have n labels, so we will plot n graphs with each label.
plt.rcParams['figure.figsize'] = (20,10)
for i in range(n_classes-1):
    y_label0 = np.array(list(map(lambda x: x[i] > 0.5, y_predicted))).astype('int')
    plt.subplot(n_classes-1, 1, i+1)
    # Plot Data
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_label0, edgecolor='k')
    plt.title("Plot for Predicted Class Sigmoid")
    plt.show()
```


![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_16_0.png)



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_16_1.png)



```python
for i in range(n_classes):
  idx = np.where(y_original==i)
  mean_score = [y_predicted[idx,j].mean() for j in range(n_classes-1)]
  print("Class Idx: {0}".format(i), mean_score)
```

    Class Idx: 0 [0.9991723, 0.0029481947]
    Class Idx: 1 [0.00020324913, 0.9997526]
    Class Idx: 2 [0.49942127, 0.48137274]


This also doesn't seem to be solvin the problem, as its classifying it into one of the classes. I ran the experiment twice and it was randomly classifying it into either.

## Single(Positive) Class Training.

In BinaryCrossEntropy when we are training for the positive label, indirectly we are also training that it doesn't belong to other class. So we need to find a mechanism such a way that we train for only Single Class(Postive), And Hence the title of the document. We have Multiple Labels but data for only single class. 

Embeddings seems to have an architecture like this, so lets visit a dual model embedding space optimiztion(idea from siamese networks).

#### Architecture
**Left Model**, is a feature generation model. Which takes input and dense layers to generate the represenatation.

**Right Model** This leg of the model learns Label Embedding.

**Comparision** This part of the model enables the learning, which tries to minimize the distance(cosine/L2/Any Distance) between the two legs of the model.


```python
# Left Side Model
input_left = Input(shape=(n_dim,), name="Input_Left")
dense_layer = Dense(n_dim,name="Dense_Left")
dense_1 = dense_layer(input_left)

#Right Side Model
model_n_classes = np.unique(y).shape[0]
input_right = Input(shape=(1,), name="Input_Right")
label_embeddings = Embedding(input_dim=model_n_classes, output_dim=n_dim, 
                             input_length=1, name="Emb_Right")
emb_right = label_embeddings(input_right)
emb_flatten = Flatten(name="Flatten")(emb_right)

# Comparision
output_euclidean = Lambda(lambda x: tf.math.reduce_euclidean_norm(x[0]-x[1], axis=[1]),
                          name='Euclidean_Distance')([dense_1, emb_flatten])

# Define Model
model = Model(inputs = [input_left, input_right], outputs=output_euclidean)
model.compile(
    loss=MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
)
model.summary()

#tf.keras.utils.plot_model(model, to_file='model_plot.png')
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Input_Right (InputLayer)        [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Input_Left (InputLayer)         [(None, 2)]          0                                            
    __________________________________________________________________________________________________
    Emb_Right (Embedding)           (None, 1, 2)         4           Input_Right[0][0]                
    __________________________________________________________________________________________________
    Dense_Left (Dense)              (None, 2)            6           Input_Left[0][0]                 
    __________________________________________________________________________________________________
    Flatten (Flatten)               (None, 2)            0           Emb_Right[0][0]                  
    __________________________________________________________________________________________________
    Euclidean_Distance (Lambda)     (None,)              0           Dense_Left[0][0]                 
                                                                     Flatten[0][0]                    
    ==================================================================================================
    Total params: 10
    Trainable params: 10
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Train Model Euclidean
X_train = [X, y]
y_train = np.zeros(X.shape[0]) # Becase we want to minimize the L2_Distance to Zero
plt.rcParams['figure.figsize'] = (20,10)
for i in range(1, 13):
    model.fit(X_train, y_train, batch_size=1, epochs=1, validation_split=0.2, verbose=0)
    plt.subplot(4, 3, i)
    # Plot Data
    new_X = dense_layer(X)
    plt.scatter(new_X[:, 0], new_X[:, 1], marker='o', c=y, edgecolor='k')
    plt.axis((-10, 10, -10, 10))

```

    WARNING:tensorflow:Layer Dense_Left is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_20_1.png)



```python
label_embeddings.weights
```




    [<tf.Variable 'Emb_Right/embeddings:0' shape=(2, 2) dtype=float32, numpy=
     array([[0.01150178, 0.02902153],
            [0.0113679 , 0.02902161]], dtype=float32)>]



What we are doing above is we are only training with Postive classes. which Says label in class is 1 is close to label in class 0. Hence bring them closer. So the losses benefits by bringing the points closer. But we want an anti force saying that these point should not be closer. 

Which is generally provided by negative samples and in our data collection mechanism we don't have those. Other alternative is random negative sampling somehting thats done in Word2Vec or other Embeddings. But that would be data corruption.

So How do I solve for this, problem at hand is the clusters are too near to be distinguished(or practically merged). Label Embeddings weights are also identical(meaning no class difference).

We can add a penality for making the embeddings for the different classes same. Something opposite to what Regularizers do. So lets introduce a anti-regularizer.

**Anti-Regularizer** basically rewards the model with far away label embeddings. But far has no limit so we would introduce a constraint on the norm of the weights of Embeddings too


```python
class AntiRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, euclidean_weight=10.0):
      '''__init__call'''
      self.eucW = euclidean_weight

    def __call__(self, W):
        return  tf.reduce_sum(tf.square(W)) - self.eucW * tf.math.reduce_sum(tf.map_fn(lambda x: self.__find_distance__(W, x) ,W))/2.0

    def __find_distance__(self, W, x):
        return tf.math.reduce_euclidean_norm(x - W)
```

The Above code calculates the distance between the inter label embeddings with negative sign. Also Weight Square is added as limiter. But we would use constraint for the sure shot limitation.


```python
# Left Side Model
input_left = Input(shape=(n_dim,), name="Input_Left")
dense_layer = Dense(n_dim,name="Dense_Left")
dense_1 = dense_layer(input_left)

#Right Side Model
model_n_classes = np.unique(y).shape[0]
input_right = Input(shape=(1,), name="Input_Right")
label_embeddings = Embedding(input_dim=model_n_classes, output_dim=n_dim, 
                             input_length=1, name="Emb_Right", 
                             embeddings_regularizer = AntiRegularizer(),
                             embeddings_constraint = tf.keras.constraints.MaxNorm(10.0))

emb_right = label_embeddings(input_right)
emb_flatten = Flatten(name="Flatten")(emb_right)

# Comparision
output_euclidean = Lambda(lambda x: tf.math.reduce_euclidean_norm(x[0]-x[1], axis=[1]),
                          name='Euclidean_Distance')([dense_1, emb_flatten])

# Define Model
model = Model(inputs = [input_left, input_right], outputs=output_euclidean)
model.compile(
    loss=MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
)
model.summary()

#tf.keras.utils.plot_model(model, to_file='model_plot.png')
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    Input_Right (InputLayer)        [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    Input_Left (InputLayer)         [(None, 2)]          0                                            
    __________________________________________________________________________________________________
    Emb_Right (Embedding)           (None, 1, 2)         4           Input_Right[0][0]                
    __________________________________________________________________________________________________
    Dense_Left (Dense)              (None, 2)            6           Input_Left[0][0]                 
    __________________________________________________________________________________________________
    Flatten (Flatten)               (None, 2)            0           Emb_Right[0][0]                  
    __________________________________________________________________________________________________
    Euclidean_Distance (Lambda)     (None,)              0           Dense_Left[0][0]                 
                                                                     Flatten[0][0]                    
    ==================================================================================================
    Total params: 10
    Trainable params: 10
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# Train Model Euclidean
X_train = [X, y]
y_train = np.zeros(X.shape[0]) # Becase we want to minimize the L2_Distance to Zero
for i in range(1, 13):
    model.fit(X_train, y_train, batch_size=1, epochs=1, validation_split=0.2)
    plt.subplot(4, 3, i)
    # Plot Data
    new_X = dense_layer(X)
    plt.scatter(new_X[:, 0], new_X[:, 1], marker='o', c=y, edgecolor='k')
```

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/indexed_slices.py:434: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "


    800/800 [==============================] - 1s 2ms/step - loss: 56.1623 - val_loss: 10.5030
    WARNING:tensorflow:Layer Dense_Left is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    800/800 [==============================] - 1s 1ms/step - loss: -5.4753 - val_loss: -17.8497
    800/800 [==============================] - 1s 1ms/step - loss: -23.5381 - val_loss: -28.5648
    800/800 [==============================] - 1s 1ms/step - loss: -31.7636 - val_loss: -34.8456
    800/800 [==============================] - 1s 1ms/step - loss: -36.7230 - val_loss: -38.6103
    800/800 [==============================] - 1s 1ms/step - loss: -39.5238 - val_loss: -40.5955
    800/800 [==============================] - 1s 1ms/step - loss: -40.9081 - val_loss: -41.5459
    800/800 [==============================] - 1s 1ms/step - loss: -41.5406 - val_loss: -41.9774
    800/800 [==============================] - 1s 1ms/step - loss: -41.8006 - val_loss: -42.1705
    800/800 [==============================] - 1s 1ms/step - loss: -41.9240 - val_loss: -42.2464
    800/800 [==============================] - 1s 2ms/step - loss: -41.9609 - val_loss: -42.3068
    800/800 [==============================] - 1s 1ms/step - loss: -42.0024 - val_loss: -42.3210



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_26_2.png)



```python
label_embeddings.weights
```




    [<tf.Variable 'Emb_Right_1/embeddings:0' shape=(2, 2) dtype=float32, numpy=
     array([[-1.11934  ,  3.9740808],
            [ 1.1585807, -4.0379505]], dtype=float32)>]



Above results seems to not merge everything to zero and keep the classes away but does it solve the task of predicting the mixed class among both?? Lets check




```python
# Calculate The Score
left_side_score = dense_layer(X)
right_side_score = label_embeddings.weights[0]
def get_prob_prediction(x):
  l2_distance = tf.math.reduce_euclidean_norm(right_side_score - x, axis=1)
  prob_score = tf.math.sigmoid(l2_distance)
  return prob_score.numpy()

y_predicted = np.array(list(map(get_prob_prediction, left_side_score)))
```


```python
# Plot Data
# Now we have two labels, so we will plot two graphs with each label.
y_label0 = np.array(list(map(lambda x: x[0] > 0.85, y_predicted))).astype('int')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_label0, edgecolor='k')
plt.title("Plot for Predicted Class Sigmoid")
plt.show()

# Now we have two labels, so we will plot two graphs with each label.
y_label1 = np.array(list(map(lambda x: x[1] > 0.85, y_predicted))).astype('int')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_label1, edgecolor='k')
plt.title("Plot for Predicted Class Sigmoid")
plt.show()
```


![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_30_0.png)



![png]({{ site.baseurl }}/assets/images/post_id_20200612/output_30_1.png)



```python
for i in range(n_classes):
  idx = np.where(y_original==i)
  mean_score = [y_predicted[idx,j].mean() for j in range(n_classes-1)]
  print("Class Idx: {0}".format(i), mean_score)
```

    Class Idx: 0 [0.6708101, 0.9994545]
    Class Idx: 1 [0.99962133, 0.6429436]
    Class Idx: 2 [0.97378165, 0.98498565]


In the Above Distribution we clearly see that the average probability score is high for both the classes for Class==2

NOTE: Although L2 Distance doesn't gurantee this probability score demarcation on sigmoid, but with a scale parameter it can always be found.


```python

```
