import tensorflow as tf
from collections import defaultdict
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing,get_data
import json 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_epochs=10
batch_size=512
learning_rate=0.01
model_dir='./../../models/mlp_models/paper_unif1'

train_datasets=[
    'Ethiopic',
    'Vai',
    'Osmanya',
    'NKo',
    'Mnist',
    'Fashion',
    'Kmnist'
]

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

init_in=tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1)
init_out=tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1)
stddev = 1 / np.sqrt(30000)
#init_in = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev)
#init_out = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev)

og_model = models.Sequential([
    layers.InputLayer(input_shape=(784,)),
    layers.Dense(30000, use_bias=False, trainable=False, kernel_initializer=init_in),
    BiasOnlyLayer(30000, activation='relu'),
    layers.Dense(10, use_bias=False, trainable=False, kernel_initializer=init_out),
    BiasOnlyLayer(10, activation='softmax'),
])

og_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

og_model.summary()

in_weights=og_model.layers[0].get_weights()[0]
in_biases=og_model.layers[1].get_weights()[0]
out_weights=og_model.layers[2].get_weights()[0]
out_biases=og_model.layers[3].get_weights()[0]
        
np.save(os.path.join(model_dir,"weights.npy"), [in_weights,out_weights])

performance_history=defaultdict(list)

for dataset in train_datasets:

    model = models.Sequential([
    layers.InputLayer(input_shape=(784,)),
    layers.Dense(30000, use_bias=False, trainable=False, kernel_initializer=init_in),
    BiasOnlyLayer(30000, activation='relu'),
    layers.Dense(10, use_bias=False, trainable=False, kernel_initializer=init_out),
    BiasOnlyLayer(10, activation='softmax'),
    ])

    model.layers[0].set_weights([in_weights]) 
    model.layers[1].set_weights([in_biases])  
    model.layers[2].set_weights([out_weights])
    model.layers[3].set_weights([out_biases])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    train_imgs,train_labels,val_imgs,val_labels=get_data(dataset)

    for i in range(num_epochs):
        
        # Train the model
        train_history = model.fit(train_imgs, train_labels,
                            epochs=1)

        curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                                train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

        val_history=model.evaluate(curr_val_imgs,curr_val_labels)

        print(f'Epoch {i+1}/{num_epochs}')
        print(f'General training accuracy: \t',train_history.history['acc'][0])

        model.save_weights((os.path.join(model_dir,'model_weights.h5')))

        if np.mean(val_history[1]) > 0.90:
            print("###############################################")
            break
    
    tot_biases=[]

    for layer in model.layers:
        if isinstance(layer, BiasOnlyLayer):
            biases = layer.get_weights()[0]  # Only one tensor: the bias
            tot_biases.append(biases)
    np.save(os.path.join(model_dir,f"{dataset}_biases.npy"), tot_biases)

    with open(os.path.join(model_dir,'perf_profiles.txt'),'w') as file:
        for task in performance_history:
            file.write(task+',')
            file.write(','.join([str(x) for x in performance_history[task]]))
            file.write('\n')
        print(performance_history)