import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing 

# Tasks:
#
#   - Even
#   - <5
#   - Prime
#   - Not Even
#   - Not <5
#   - Not Prime

all_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']

num_epochs=1000
batch_size=64
learning_rate=0.005
num_neurons_1=10000
num_neurons_2=5000
num_neurons_3=5000
num_neurons_4=5000
train_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']
model_dir='./../../models/mlp_models/odd_notodd_firsthalf_notfirsthalf_prime_full_10-32_005'

# Custom Bias-Only Layer
class BiasOnlyLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, initializer='zeros', **kwargs):
        super(BiasOnlyLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.initializer = initializer
    
    def build(self, input_shape):
        # Create only bias, no weights
        self.bias = self.add_weight(
            name='bias', 
            shape=(self.units,),
            initializer=self.initializer,
            trainable=True
        )
        super(BiasOnlyLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Just add bias
        output = inputs + self.bias
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)
        
        return output

uniform_initializer=tf.keras.initializers.RandomUniform(minval=-1,maxval=1)
normal_initializer=tf.keras.initializers.RandomNormal()

model = models.Sequential([
    layers.InputLayer(input_shape=(784+len(all_tasks),)),
    
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),
    #layers.Dense(1000, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    #BiasOnlyLayer(1000, activation='relu'),

    # First layer without bias
    #layers.Dense(num_neurons_1, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_1, activation='relu'),
    
    # First layer without bias
    #layers.Dense(num_neurons_2, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_2, activation='relu'),

    # First layer without bias
    #layers.Dense(num_neurons_3, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_3, activation='relu'),
    
    #First layer without bias
    #layers.Dense(num_neurons_4, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_4, activation='relu'),
    
    # Output layer without bias
    #layers.Dense(1, use_bias=False, trainable=False, kernel_initializer=normal_initializer), 
    
    # Custom bias-only layer for output
    #BiasOnlyLayer(1, activation='sigmoid')

    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(32, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(1, use_bias=True, trainable=True, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Load data
train_set,_=datasets.mnist.load_data()
train_imgs,train_labels=train_set

# Split into training and validation sets
validation_split = 0.2
split_index = int(len(train_imgs) * (1 - validation_split))
train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

# Preprocess data
train_imgs,train_labels=preprocessing(imgs=train_imgs,labels=train_labels,
                                      train_tasks=train_tasks,all_tasks=all_tasks)

for i in range(num_epochs):

    # Train the model
    train_history = model.fit(train_imgs, train_labels,
                        epochs=1,
                        batch_size=batch_size)

    val_history={}

    for task in train_tasks:

        curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                            train_tasks=[task],all_tasks=all_tasks)

        val_history[task]=model.evaluate(curr_val_imgs,curr_val_labels)

    print(f'Epoch {i+1}/{num_epochs}')
    print(f'General training accuracy: \t',train_history.history['acc'][0])
    for task in train_tasks:
        print(f'{task} validation accuracy: \t', val_history[task][1])

    os.makedirs(model_dir,exist_ok=True)
    model.save_weights((os.path.join(model_dir,'model_weights.h5')))