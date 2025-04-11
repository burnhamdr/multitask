import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_tasks=['odd','firsthalf','prime']
num_epochs=10
batch_size=64
learning_rate=0.01
num_neurons_1=20000
num_neurons_2=5000
num_neurons_3=2500
train_tasks=['odd','firsthalf','prime']
model_dir='./../../models/mlp_models/odd_firsthalf_prime_20000_01'

# Load data
train_set,_=datasets.mnist.load_data()
train_imgs,train_labels=train_set

# Split into training and validation sets
validation_split = 0.2
split_index = int(len(train_imgs) * (1 - validation_split))
train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

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
    layers.Dense(num_neurons_1, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    BiasOnlyLayer(num_neurons_1, activation='relu'),

    # First layer without bias
    #layers.Dense(num_neurons_2, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_2, activation='relu'),

    # First layer without bias
    #layers.Dense(num_neurons_3, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    
    # Custom bias-only layer for first hidden layer
    #BiasOnlyLayer(num_neurons_3, activation='relu'),
                                        
    # Output layer without bias
    layers.Dense(1, use_bias=False, trainable=False, kernel_initializer=normal_initializer), #, kernel_initializer=tf.constant_initializer([1.,1.])
    
    # Custom bias-only layer for output
    BiasOnlyLayer(1, activation='sigmoid')
])

model.load_weights(os.path.join(model_dir,'model_weights.h5'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

val_history={}

for task in train_tasks:

    curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                        train_tasks=[task],all_tasks=all_tasks)

    val_history[task]=model.evaluate(curr_val_imgs,curr_val_labels)

for task in train_tasks:
    print(f'{task} validation accuracy: \t', val_history[task][1])

"""
test_rule_inputs={'Firsthalf':[1,0,0,0,0],
                  'Not firsthalf':[0,1,0,0,0],
                  'Under 3':[0,0,1,0,0],
                  'Not under 3':[0,0,0,1,0],
                  'Under 7':[0,0,0,0,1],
                  '-firsthalf':[-1,0,0,0,0],
                  '-under3':[0,0,-1,0,0],
                  '-under7':[0,0,0,0,-1],
                  'not firsthalf-firsthalf+under7':[-1,1,0,0,1],
                  'not firsthalf+firsthalf+under7':[1,1,0,0,1],
                  'not under3-under3+under7':[0,0,-1,1,1],
                  'not under3+under3+under7':[0,0,1,1,1],
                  'under3-not under3+under7':[0,0,1,-1,1],
                  'not first half as firsthalf+not under3-under3':[1,-1,0,1,0],
                  'not first half as firsthalf+not under3+under3':[1,1,0,1,0]}
"""
"""
'Is the number Odd?':[1,0,0,0,0,0],
                    'Is the number in the first half?':[0,1,0,0,0,0],
                    'Is the number prime?':[0,0,1,0,0,0],
                    'Is the number not Odd?':[0,0,0,1,0,0],
                    'Is the number not in the first half?':[0,0,0,0,1,0],
                    'Is the number not prime':[0,0,0,0,0,1],
                    'Is the nubmer Odd? \nAND\n Is the number in the first half?':[1,1,0,0,0,0],
                    'Is the number in the first half? \nAND\n Is the number prime?': [0,1,1,0,0,0],
                    'Is the number Odd? \nAND\n Is the number prime?':[1,0,1,0,0,0],
                    'Odd-firsthalf':[1,-1,0,0,0,0],
                    'Firsthalf-prime':[0,1,-1,0,0,0],
                    'Odd-prime':[1,0,-1,0,0,0],
                    '-odd':[-1,0,0,0,0,0],
                    '-firsthalf':[0,-1,0,0,0,0],
                    '-prime':[0,0,-1,0,0,0],
                    'Is the number Odd? \nAND\n Is the number in the first half? \nAND\n Is then number prime?':[1,1,1,0,0,0],
                    'Compositional attempt at NOT ODD (using half)':[1,-2,0,0,1,0],
                    'Compositional attempt at NOT ODD (using prime)':[1,0,-2,0,0,1],
                    '(-half)-half+odd':[1,-1,0,0,1,0],
                    '(-prime)-prime+odd':[1,0,-1,0,0,1],
                    'half+(-half)+odd':[1,1,0,0,1,0],
                    'prime+(-prime)+odd':[1,0,1,0,0,1],
                    '(-half)-half':[0,-1,0,0,1,0],
                    '(-prime)-prime':[0,0,-1,0,0,1],
                    '-(-odd)':[0,0,0,-1,0,0]
"""
test_rule_inputs={
                    'Is the number Odd?':[1,0,0],
                    'Is the number in the first half?':[0,1,0],
                    'Is the number prime?':[0,0,1],
                    'Is the nubmer Odd? \nAND\n Is the number in the first half?':[1,1,0],
                    'Is the number in the first half? \nAND\n Is the number prime?': [0,1,1],
                    'Is the number Odd? \nAND\n Is the number prime?':[1,0,1],
     
}

for text in test_rule_inputs:

    print('Preparing graph for rule input [',",".join([str(num) for num in test_rule_inputs[text]]),']')
    formatted_imgs=[]

    for img in val_imgs:

            # Build input by concatenating flattened image and rule input
            flattened_img=img.reshape(784)/255.
            rule_input=np.array(test_rule_inputs[text])
            final_input=np.concatenate([flattened_img, rule_input])
            formatted_imgs.append(final_input)

    formatted_imgs=np.array(formatted_imgs)

    out = model.predict(formatted_imgs)
    out = np.array([1 if x>0.5 else 0 for x in out])

    cm = confusion_matrix(val_labels, out)[:,:2]
    new_cm=[]

    for i,row in enumerate(cm):
        tot=sum(row)
        new_cm.append([x/tot*100 for x in row])

    plt.figure(figsize=(8, 8))
    sns.heatmap(new_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=[i for i in range(10)])
    plt.xlabel("Prediciton confidence")
    plt.ylabel("Number")
    plt.title(text)
    plt.savefig(os.path.join(model_dir,f"{'_'.join([str(i) for i in test_rule_inputs[text]])}_result.png"))
    plt.clf()