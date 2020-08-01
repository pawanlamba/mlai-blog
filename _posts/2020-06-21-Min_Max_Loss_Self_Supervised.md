---  
layout: post  
title:  "Min-Max Loss, Self Supervised Classification"  
author: pawan  
categories: [ Loss Function, Deep Learning, Self Supervised Learning ]  
image: assets/images/post_id_20200620/self-supervised-confusion-matrix.svg  
featured: true  
---  
  
We left last post [Min Max Loss Classification Example]({{site.baseurl}}/Min_Max_Loss_Revisting_Classification_Loss/) on the promise to demonstrate a self supervised classification example.  
  
**What is Self Supervised Classification.**  
  
Self Supervised Classification is a task of learning classes from the given data, without any explicitly tagged data. There has been some recent research into domain, [SimCLR](https://arxiv.org/abs/2002.05709), [SimCLRv2](https://arxiv.org/abs/2006.10029), [Learning to Classify Images Without Labels](https://arxiv.org/abs/2005.12320).   
  
Continuing from where we left Min-Max Loss defines classification as minimizing the distance between the class label and sample and maximizing the inter-class distance. With a lack of class labels for self-supervised learning, we would have the liberty to maximize the inter-class difference upfront, by adopting one-hot encoding.  
  
One hot encoding is the maximum entropy representation of $n$ dimensional $n$ vectors, constrained on the unit norm and positive values only[or probability space].   
We can choose any distance metric and this would hold true. The below image summarizes this.  
  
![Unit Norm Maxima](https://upload.wikimedia.org/wikipedia/commons/d/d4/Vector-p-Norms_qtl1.svg).  
  
### Self Supervised Loss  
  
In absence of class labels, we define a new loss, on the lines of Min-Max, We do acknowledge for self-supervised/un-supervised scenarios it is the same as contrastive loss or Negative Sampling.  
  
For Self-supervised classification, we need a measure of distance defined over sample space $X$. For numerical data, it is possible, for images and text we can learn embeddings to achieve this.  
  
Let $C$ be set of classes we want our data to be classified into, for a sample $x \in X$ let the $F_x$ denote the set of Farthest Samples from $x$, and $N_x$ denote the set of nearest neighbors of the $x$. Let $\psi_{x,c}$ be a function we want to learn which defines the distribution that $x$ belongs to class $c$. $D$ is the distance metric between two probability distributions.  
  
We expect the nearest neighbors belong to the same class and farthest neighbors don't belong to the same class. Also maximizing the classes covered.  
  
$$min \sum_{x \in X}\sum_{n_x \in N_x}D(\psi(x),\psi(n_x))$$  
  
$$max \sum_{x \in X}\sum_{f_x \in F_x}D(\psi(x),\psi(f_x))$$  
  
$$max \sum_{\forall C} D1(c_i, c_j)$$  
  
it is very close to the SCAN loss discussed in "Learning to Classify Images Without Labels", with an additional loss term for far off samples, and we use softmax for a soft assignment which assures probability consistency. This can be seen as an extension of the work.  
  
For the synthetic example below we use, distance metric $D$ as dot product, and for $D1$ we use is the entropy metric.  
  
Hence the loss for our example can be termed as SCANv2 loss.  
  
  
$$min \sum_{x \in X}\sum_{n_x \in N_x}\psi(x).\psi(n_x)) - \sum_{x \in X}\sum_{f_x \in F_x}\psi(x).\psi(f_x) + \lambda \sum_{\forall C} \psi^clog\ \psi^ c $$  
  
$$with\  \psi^c = \frac{1}{|X|} \sum_{x \in X} \psi(x)\ for\ c$$  
  
  
**Disclaimer**  
  
Like all clustering losses, it depends upon the density of the near point and then density gradient when switching from one class to another. So for tasks, we need to have embeddings that can establish the concept of near and far on the space on the dimension we want the classification on. There are existing ways/concepts to train embeddings this way. Including the SCAN loss paper discusses one for semantic clustering.  
  
  
### Synthetic Example  
  
**Data Setup**  
 - We randomly create 10 blobs of data with 100 samples each.   
 - Then for each sample in the above 1000 samples, we create 10 nearest neighbors.  
 - We also find 900 farthest neighbors. (we can resort to negative sampling from the list too)  
 - For the nearest neighbor, we add cls_tag as 1 and far off neighbor we add cls_tag as 0. These values represent dot product values we want between neighbors and far off samples. Zero signifies orthogonal class, 1 signifies the same class.  
 - Now we have 3 arrays of 910K samples each. "X\_train" [is X repeated 910 times], "X\_other" is array of near and far off samples, "cls\_all" is array of distance metric values [1,0].  
  
 Please find below the distribution of the data.  
 ![Data Blob For Self Supervised]({{site.baseurl}}/assets/images/post_id_20200620/self-supervised-data-blob.svg)  
  
   
```python  
import tensorflow as tf  
import numpy as np  
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Flatten  
from tensorflow.keras.layers import Layer, Dot, Activation, Softmax, BatchNormalization  
from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy  
  
# Define AntiRegularizer for maximizing Entropy  
class AntiRegularizerEntropy(tf.keras.regularizers.Regularizer):  
  
 def __init__(self, strength=1.0):  
  self.strength = strength  
  
 def __call__(self, x):  
 prob_distribution = tf.math.reduce_mean(x, axis=0)  
 entropy = prob_distribution * tf.math.log(prob_distribution)/tf.math.log(2.0) return (self.strength * tf.math.reduce_sum(entropy))  
  
  
# Feature Encoder  
def get_feature_encoder():  
 feat_inp = Input(shape=(n_dim,), name="feat_inp")  
 x = Dense(n_classes, name="dense_1",   
 kernel_initializer= 'random_uniform',   
 activation='relu')(feat_inp)  
 feat_encoding = Dense(n_classes, name="dense",   
 kernel_initializer= 'random_uniform',   
 activation='softmax',  
  activity_regularizer = AntiRegularizerEntropy(10.0))(x) return Model(feat_inp, feat_encoding)  # Define Model  
x_in = Input(shape=(n_dim,), name="x_in")  
x_oth = Input(shape=(n_dim,), name="x_oth")  
  
feat_encoder = get_feature_encoder()  
x_in_enc = feat_encoder(x_in)  
x_oth_enc = feat_encoder(x_oth)  
  
# Comparision  
x_in_oth_dist = Dot(axes=1, normalize=True)([x_in_enc, x_oth_enc])  
  
# Define Model  
model = Model(inputs = [x_in, x_oth], outputs=x_in_oth_dist)  
model.compile(  
  loss=[MeanSquaredError(), MeanSquaredError()],  
  optimizer=tf.keras.optimizers.Adam(),  
)  
model.summary()  
  
model.fit([X_train, X_other], cls_all, epochs=100, batch_size=2048, shuffle=True)  
```  
     Epoch 100/100  
 450/450 [==============================] - 1s 3ms/step - loss: -0.0271  
**Confusion Matrix**  
  
We plot the confusion matrix below, We expect it to be a diagonalizable matrix, with the order of columns changed. Any deviation from that is termed as inaccuracy.  
  
```python  
y_predicted = tf.math.argmax(feat_encoder.predict(X),axis=1)  
confusion_matrixtf.math.confusion_matrix(y, y_predicted)  
```  
Lets plot confusion matrix with some columns interchanged.  
  
![Self Supervised Confusion Matrix]({{site.baseurl}}/assets/images/post_id_20200620/self-supervised-confusion-matrix.svg)  
  
We find two classes have some samples interchanged. Let us plot them and check.  
  
Below is the classification results plotted.  
![Self Supervised Classification Plot]({{site.baseurl}}/assets/images/post_id_20200620/self-supervised.svg)  
   
We can see two classes are partitioned on a different axis as intended, Also we have seen while running experiments, when classes are fuzzy and connected together, then classification is often erroneous. But so is it in other algorithms.  
  
### Next Steps  
The algorithm above demonstrates 98.6% accuracy in a single run since it also suffers from initialization bias, we can run multiple iterations of it to improve accuracy like other clusterings algorithms if it gets stuck in local minima.  
  
Further, we can include other heuristics to improve classification accuracy. By including neighbors and excluding far off point we followed a simple distance-based approach.    
Because of the lack of computing power I have demonstrated results on the synthetic data. For other tasks like images/text/graphs. we can learn embeddings first and follow the steps as above. Although it is advised that embeddings have some concept of density variation between similar and dissimilar semantic on which we cluster.