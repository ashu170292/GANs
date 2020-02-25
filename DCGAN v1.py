
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Input
import time


# In[ ]:


mnist_images = tf.keras.datasets.mnist.load_data()


# In[ ]:


train_mnist_images = mnist_images[0][0]
train_mnist_labels = mnist_images[0][1]
test_mnist_images = mnist_images[1][0]
test_mnist_labels = mnist_images[1][1]


# In[ ]:


train_mnist_images[0].shape # 28 x 28


# In[ ]:


# BUFFER_SIZE = 60000  
# BATCH_SIZE = 256  #Num image in one batch
train_mnist_images = train_mnist_images.reshape(train_mnist_images.shape[0], 28, 28, 1).astype('float32')
train_mnist_images = (train_mnist_images - 127.5) / 127.5 
train_mnist_images = train_mnist_images.reshape(train_mnist_images.shape[0],
                                                train_mnist_images.shape[1]*train_mnist_images.shape[2])
#train_dataset = tf.data.Dataset.from_tensor_slices(train_mnist_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[ ]:


noise_dim =100
image_dim = 784
#random_noise = np.random.normal(size=noise_dim)

def generator_model():
    model_gen = Sequential()
    model_gen.add(Dense(50, input_dim=noise_dim))
    model_gen.add(Activation('relu'))
    model_gen.add(Dense(30))
    model_gen.add(Activation('relu'))
    model_gen.add(Dense(50))
    model_gen.add(Activation('relu'))
    model_gen.add(Dense(image_dim,activation='tanh'))
    return model_gen


def discriminator_model():
    model_disc = Sequential()
    model_disc.add(Dense(50, input_shape=(784,)))
    model_disc.add(Activation('relu'))
    model_disc.add(Dense(30))
    model_disc.add(Activation('relu'))
    model_disc.add(Dense(50))
    model_disc.add(Activation('relu'))
    #model_disc.add(Flatten())
    model_disc.add(Dense(1,activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model_disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model_disc 

model_test = discriminator_model()
model_test.summary()


# In[5]:


np.random.normal(size=(4,3))


# In[ ]:


#adversarial training

def adversarial_training(G, D):
    D.trainable = False
    noise_vec = Input(shape=(noise_dim,))
    g_z = G(noise_vec)
    d_g_z = D(g_z)
    gan = Model(noise_vec,d_g_z)
    opt = Adam(lr=0.1)
    gan.compile(loss='binary_crossentropy',optimizer = opt , metrics = ['accuracy'])
    return gan



G = generator_model()
D = discriminator_model()

gan = adversarial_training(G,D)
    
    


# In[3]:


import numpy as np
np.random.randint(0, 10, 4)


# In[ ]:


####GAN algo

loss_gen_final =[]
def train_the_GAN(epochs,batch_size,image_data):
    #pass
    for i in range(epochs):
        random_noise = np.random.normal(0,1,size=(batch_size,noise_dim))
        fake_images = G.predict(random_noise)
        print (type(fake_images))
        tr_idx = np.random.randint(0, image_data.shape[0], batch_size) #revisit
        imgs = image_data[tr_idx]
        
        data_overall = np.concatenate(imgs,fake_images)
        
        labels_real = np.ones(batch_size)
        label_fake = np.zeros(batch_size)
        labels_overall = np.concatenate(labels_real,labels_fake)
        
        discriminator.trainable = True
        
        discriminator.train_on_batch(data_overall,labels_overall)
        
        loss_disc = gan.evaluate(data_overall,labels_overall,verbose=1) 
        
        random_noise = np.random.normal(size=(batch_size,noise_dim))
        labels_gen =np.ones(batch_size)
        
        discriminator.trainable = False
        gan.train_on_batch(random_noise,labels_gen)
        
        loss_gen = gan.evaluate(random_noise,labels_gen,verbose=1)
        
        loss_gen_final.append(loss_gen)
        
    return loss_gen_final
        
        
        
        
    
    
test_mod = train_the_GAN(10,25,train_mnist_images)
        
        
        
        
        
        
        
        

