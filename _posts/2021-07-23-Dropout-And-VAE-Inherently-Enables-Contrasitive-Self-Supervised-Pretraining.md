---  
layout: post  
title:  "Dropout And VAE Inherently Enables Contrastive Self Supervised Pretraining"  
author: pawan  
categories: [Deep Learning, Self Supervised Learning, Pretraining, Dropout, VAE ]  
image: assets/images/post_id_20210724/Dropout.svg  
featured: true  
---  

Self-Supervised training is a eureka concept, where machines don't need labels to learn concepts, It started with a lack of tagged data and solution being self-supervised training. However, in recent times for supervised tasks too, it is advisable to pre-train your encoder on a self-supervised task. Pretraining happens to learn the world model of things and hence can generalize and improve the performance of supervised tasks.  
  
There is enormous evidence available that self-supervised pretraining task improves the performance and converge time for the downstream supervised task. In the tabular dataset models like TabNets, TaBERT etc we find proposals for self-supervised pretraining. One of the reasons that recent neural network-based methods win over the tree-based methods is self-supervised pretraining.  
  
Self-Supervised pretraining provides a great advantage, but its an additional task, ML scientist needs to train on. Also, the advantage is well recorded in low data regimes, but the advantage diminishes with increasing samples. To solve this, can we devise methods to fulfil the objective in the main training loop itself. Methods exist where one can add pretraining loss as multi-task objectives, but that too is additional flops. Is there a method where we don't have to add extra flops and yet embed pretraining somehow? we discuss below how that's achievable or rather embedded in supervised training
  
> In this piece we argue that adding a DropOut layer or VAE layer is equivalent to minimizing an auxiliary self-supervised loss along with supervised loss in a multi-epoch system. How these stochastic layers act as data augmentation techniques for self-supervised method and label acts as our $Coded\ Protype\ Vectors$ from **SWaV** in a multi-epoch/oversampled training.  
  
### Dropout and VAE.  
Both dropout and VAE are understood to add to the generalisation and robustness of the model, but the exact mechanism of working is not very clear.   
> We propose that one of the reasons for the success of Dropout and VAE layers is the inherent self-supervised learning they enable in the training loop.  
  
DropOut masks random neurons and resembles the masking technique used in BERT based self-supervised learning.  Masks are studied well in vision tasks and are widely used, masking mechanism in images resembles working of as 1D,2D..3D SpatialDropout layers.  
  
VAE adds Gaussian noise to a layer that can be conceived of as a data augmentation technique in latent space. Simultaneously also resembles the corruption process in self-supervised learning.   
  
As VAE also finds application as a layer for data augmentation in latent space, we can conveniently comprehend that data augmentation techniques and self-supervised learning produce the same impact on learning in a multi-epoch supervised system. It's just two different ways to achieve the same goal.   
  
![Self Supervised Training]({{site.baseurl}}/assets/images/post_id_20210724/SelfSupervised.svg)  
  
### Self Supervised Pre-Training  
The contrastive self-supervised pretraining process can be understood as a reconstruction of random mask/corruption on feature vector $X$. Where corruption can be introduced by Dropout, VAE or any other layer.  
  
In the training setup we have two encoders $encoder(m)$ and $encoder(u)$ with weights shared. which encodes two inputs $X_m$ and $X_u$ into representations $m$ and $u$ respectively. Where $X_m$ is defined as $X * mask$. Then we have a contrastive loss that compares $m,u$ and trains back our encoder with the gradients.   
  
In other variants of self-supervised training instead of directly comparing $\<m.u\>$, one can coerce them to the same value/clusters. The Idea is to maximise agreement between representation. SWaV is one example where we want to coerce both the legs of an encoder to the same Coded Prototype Vector.  
  
### Supervised Learning.  
While in our supervised setup we have a network/encoder that projects feature vector $X$ into representation $u$ and then supervised loss coerce that to label $y$, or maximizes agreement with the label, as shown in the picture below  
  
![Supervised Training]({{site.baseurl}}/assets/images/post_id_20210724/Supervised.svg)  
  
### Supervised Training With Dropout.  
**Single Epoch Training**  
  
Adding a dropout layer before our encoder network in a supervised setup essentially generates a masked vector $X_m$ from feature vector $X$. Similar to the first step of self-supervised training, the network now will generate representation $m$ of masked vector $X_m$, that by supervised loss will be coerced to label $y$.  
  
![Supervised Training]({{site.baseurl}}/assets/images/post_id_20210724/DropoutSupervised.svg)  
  
**Multi Epoch Training**  
  
We need to observe and compare the dynamics of training for the same sample when we train this network for $n$ epochs.   
  
In an $n$ epoch an input sample $X_i$ will be repeated multiple times across epochs. Let us compare training dynamics of instances of $X_i$ across epochs,  
  
- **Corruption**: We observe that the masked vector due to the stochastic nature of the dropout layer are generated differently,   
hence $$X_{m1} \ne X_{m2} \ne ... \ne X_{mn}$$  
- **Representation**: Correspondingly representation generated by encoder for all $X_m's$ are $m_1, m_2 ... m_n$.    
  
![Supervised Training]({{site.baseurl}}/assets/images/post_id_20210724/Dropout.svg)  

- **Maximize Agreement**: Representations are compared with label $y_i$, using the loss function across epochs.
  
Since representation will be compared with the same label $y_i$ there must exist an agreement between $m_1, m_2, ... m_n$ by the design of the training setup. On completion of the training loop, it would be right to assume that  
  
$$m_1 \approx m_2 \approx ... \approx m_n$$  
  
On careful reconsideration, the training process resembles the process of self-supervised training, where we are corrupting the input $X_i$ and then trying to minimize distance or maximize agreement between their representations $m_1, m_2, ... m_n$.  
  
Multi-epoch here is just for explanation, to make it more explicit we can up-sample the records and zip them in the same batch. We have used dropout as a layer of corruption, another layer like VAE can replace dropout and without the loss of generality. 

We can conclude that while minimizing the supervised loss, we are simultaneously minimizing the auxiliary self-supervised loss.