import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing 

# Tasks:
#
#   - Even-Odd
#   - <5->=5
#   - Prime-Nonprime

all_tasks=['odd','firsthalf','prime']

num_epochs=10
batch_size=64
learning_rate=0.001
train_tasks=['odd','firsthalf','prime']
model_dir='./../../models/mlp_models/odd_firsthalf_prime5'

model = models.Sequential([
    layers.InputLayer(input_shape=(784+len(all_tasks),)),
    #layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
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

