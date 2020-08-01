---
layout: post
title:  "Min-Max Loss, Revisiting Classification Losses"
author: pawan
categories: [ Loss Function, Deep Learning, Classification ]
image: assets/images/post_id_20200615/training_setup.svg
featured: true
---

In continuation to my [Partial Tagged Data Classification]({{site.baseurl}}/Multi_Label_Single_Class_Learning_Classifying_Partially_Tagged_Data) post, We formulate a generic loss function applicable to all task(classification, metric learning, clustering, ranking, etc)

Traditionally we had classification losses like CrossEntropy, Log-Likelihood, MSE, MAE, etc. But recently a more general approach for metric learning is famous like [contrasitive loss, Chopra Et al.](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), [Triplet Loss, Weinberger and Saul](http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf), Margin Loss, etc. Recently [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) showed that metric learning can overpower classification losses.

Traditional Classification losses like cross-entropy, log-likelihood modulates feature space representation and are linear classifiers over the resultant space. Hence often assume a sample belongs to one of the available classes. which makes it unsuitable for tasks like recommendations, missing link predictions, ranking documents, Embeddings, etc.[Reason why metric loss is picking up]. Where samples may belong to other classes too.

Metric learning losses assume the availability of the contrastive data or rely on the mechanism to carefully sample negative labels. Which could lead to in-efficiencies in the learning process?
 
We explore a new loss that is free from negative sampling. We propose a Min-Max loss which minimizes a metric between sample and class and maximizes intra-class distance(class separability). Let $C$ be set of classes and $X$ be set of samples.

$$min \sum_{x\in X, c \in C}D(x,c)$$

$$max \sum_{\forall C} D(c_i,c_j)$$

where sample $x$ is tagged to class $c$.

Intuitively we are pushing samples close to its class's learned representation and pushing all classes away from each other to maintain separability.

For simplifying it to a minimization problem we can rewrite it as 

$$min \sum_{x\in X, c \in C}D(x,c) - \sum_{\forall C} D(c_i,c_j)$$

We are free to choose any distance convex metric like L2, L1, Cosine Similarity, KL-Divergence, etc.

For non-probabilistic metrics, we need to have a constraint on the $D(c_i,c_j)$ as it can not converge.

$$min \sum_{x\in X, c \in C}D(x,c) - \sum_{\forall C} D(c_i,c_j)$$

$where$

$$\sum_{\forall C} D(c_i,c_j) < K ... Norm Constraint$$

## Training Setup

**Supervised Setup**<br/>

For supervised training, we can set up this as either loss from the label embedding layer with constraints in TensorFlow. Something we did in the [Partial Tagged Data Classification]({{site.baseurl}}/Multi_Label_Single_Class_Learning_Classifying_Partially_Tagged_Data) post. This could be costly when we have a large number of classes. 

An alternate implementation is with negative sampling(not among samples but classes). 

Setup Diagram Below

![Training SetUp]({{site.baseurl}}/assets/images/post_id_20200615/training_setup.svg)

Where
- Label Encoder is a Label Embedding Layer.
- Feat Encoder is a neural network for feature transformations.
- We Feed positive(1) or negative label(-1) on the respective cosine losses. 
For other distance metrics, we can choose accordingly.

**Supervised Experiment**
We keep the data setup similar to the previous post and train the model using the below code.

**Goal**: Classify common class as part of both classes.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Flatten, Dot, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy

# Feature Encoder
feat_inp = Input(shape=(n_dim,), name="feat_inp")
dense_layer = Dense(n_dim, name="dense")
feat_encoding = dense_layer(feat_inp)

# Label Encoder
model_n_classes = np.unique(y).shape[0]
label_embeddings = Embedding(input_dim=model_n_classes, output_dim=n_dim, 
                             input_length=1, name="emb_label")

# Class Embeddings
cls_inp = Input(shape=(1,), name="cls_inp")
emb_cls = label_embeddings(cls_inp)
cls_flatten = Flatten(name="cls_flatten")(emb_cls)

# OTH_Class Embeddings
oth_cls_inp = Input(shape=(1,), name="oth_cls_inp")
emb_oth_cls = label_embeddings(oth_cls_inp)
oth_cls_flatten = Flatten(name="oth_cls_flatten")(emb_oth_cls)

# Comparision
cls_feat_output = Dot(axes=1, normalize=True, name='cls_feat_output')([feat_encoding, cls_flatten])
cls_oth_output = Dot(axes=1, normalize=True, name='cls_oth_output')([oth_cls_flatten, cls_flatten])

# Define Model
model = Model(inputs = [feat_inp, cls_inp, oth_cls_inp], outputs=[cls_feat_output, cls_oth_output])
model.compile(
    loss=[MeanSquaredError(), MeanSquaredError()],
    optimizer=tf.keras.optimizers.Adam(),
)
model.summary()

## Prepare Data
import random
uniq_classes = np.unique(y)
oth_cls_array = np.array([random.choice(uniq_classes[uniq_classes != i]) for i in y_original])
input_seq = [X, y, oth_cls_array]
output_seq = [np.ones(n_samples), -np.ones(n_samples)]

##Train Model
model.fit(input_seq, output_seq, batch_size=1, epochs=10, validation_split=0.2)
```

    Epoch 10/10
    800/800 [==============================] - 1s 1ms/step - loss: 0.9283 - cls_feat_output_loss: 0.1606 - cls_oth_output_loss: 0.7677 - val_loss: 0.7572 - val_cls_feat_output_loss: 0.1703 - val_cls_oth_output_loss: 0.5869
    
    <tensorflow.python.keras.callbacks.History at 0x7f87c5a58150>


```python
# Calculate The Score and Label
feature_encoding = dense_layer(X).numpy()
label_encoding = label_embeddings.weights[0].numpy()
def get_score(x):
  dot_product =  [5.0*np.dot(this_label_encoding, x)/(np.linalg.norm(x)*np.linalg.norm(this_label_encoding))
                  for this_label_encoding in label_encoding]
  return tf.math.sigmoid(dot_product) > 0.5

y_label = np.array(list(map(get_score, feature_encoding)))

# Plot Data
plt.rcParams['figure.figsize'] = (20,10)

# Plot Class=0
plt.subplot(1, 2, 1)
new_X = dense_layer(X)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_label[:,0], edgecolor='k')
plt.title("Plot for Class=0")

# Plot Class=1
plt.subplot(1, 2, 2)
new_X = dense_layer(X)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_label[:,1], edgecolor='k')
plt.title("Plot for Class=1")

```

![SupervisedPlot]({{site.baseurl}}/assets/images/post_id_20200615/SupervisedPlot.svg)

From the charts, we can deduce that the common class in original data(class==2) is included in both label boundaries.
hence can say the goal is accomplished. We have trained a classification model using our loss and successfully avoided wrongly tagging mixed class. 

**Self-Supervised Setup**<br/>
It demands a section of its own hence we will cover how to use the min-max loss for self-supervised learning.