import tensorflow as tf
from tensorflow.keras import layers, models, losses
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing 
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Tasks:
#
#   - Even
#   - <5
#   - Prime
#   - Not Even
#   - Not <5
#   - Not Prime

all_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']

num_epochs=5
batch_size=64
learning_rate=0.005
num_neurons_1=1000
num_neurons_2=5000
num_neurons_3=5000
num_neurons_4=5000
train_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']
model_dir='./../../models/mlp_models/bah_prova_auto'

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
    layers.Dense(512, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(256, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(10, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(256, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(512, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(784+len(all_tasks), use_bias=True, trainable=True, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=losses.MeanSquaredError(),
              metrics=['mean_squared_error'],)

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
trial=train_imgs[1,:]
train_imgs,train_labels=preprocessing(imgs=train_imgs,labels=train_labels,
                                      train_tasks=train_tasks,all_tasks=all_tasks, class_training=True)

for i in range(num_epochs):

    # Train the model
    train_history = model.fit(train_imgs,train_imgs,
                        epochs=1,
                        batch_size=batch_size)

    curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                        train_tasks=[],all_tasks=all_tasks, class_training=True)

    val_history=model.evaluate(curr_val_imgs,curr_val_imgs)

    print(f'Epoch {i+1}/{num_epochs}')
    print(train_history.history)
    print(f'General training accuracy: \t',train_history.history['mean_squared_error'][0])

    print(f'Classification validation accuracy: \t', val_history[1])

    #os.makedirs(model_dir,exist_ok=True)
    #model.save_weights((os.path.join(model_dir,'model_weights.h5')))

plt.imshow(trial, cmap='gray')
plt.savefig('a.png')
flattened_img=trial.reshape(784)/255.
rule_input=np.zeros(len(all_tasks))
final_input=np.concatenate([flattened_img, rule_input])
formatted_img=np.array(final_input)
formatted_img=formatted_img.reshape(1,len(formatted_img))
plt.imshow(model.predict(formatted_img)[:,:-5].reshape((28,28)),cmap='gray')
plt.savefig('b.png')

# Creating new model
new_model = models.Sequential([
    layers.InputLayer(input_shape=(784+len(all_tasks),)),
    layers.Dense(512, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(256, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(10, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(1, use_bias=True, trainable=True, activation='sigmoid'),
])

new_model.layers[0].set_weights(model.layers[0].get_weights())
new_model.layers[1].set_weights(model.layers[1].get_weights())
new_model.layers[2].set_weights(model.layers[2].get_weights())

new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

new_model.summary()

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
                                      train_tasks=train_tasks,all_tasks=all_tasks, class_training=False)

for i in range(num_epochs):

    # Train the model
    train_history = new_model.fit(train_imgs, train_labels,
                        epochs=1,
                        batch_size=batch_size)

    val_history={}

    for task in train_tasks:

        curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                            train_tasks=[task],all_tasks=all_tasks, class_training=False)

        val_history[task]=new_model.evaluate(curr_val_imgs,curr_val_labels)

    print(f'Epoch {i+1}/{num_epochs}')
    print(f'General training accuracy: \t',train_history.history['acc'][0])

    training_finished=True

    for task in train_tasks:
        if val_history[task][1]<0.95: training_finished=False
        print(f'{task} validation accuracy: \t', val_history[task][1])

    os.makedirs(model_dir,exist_ok=True)
    new_model.save_weights((os.path.join(model_dir,'model_weights.h5')))

    if training_finished: break

test_rule_inputs={'Is the number Odd?':[1,0,0,0,0],
                    'Is the number in the first half?':[0,1,0,0,0],
                    'Is the number prime?':[0,0,1,0,0],
                    'Is the nubmer Odd? \nAND\n Is the number in the first half?':[1,1,0,0,0],
                    'Is the number in the first half? \nAND\n Is the number prime?': [0,1,1,0,0],
                    'Is the number Odd? \nAND\n Is the number prime?':[1,0,1,0,0],
                    'Is the number Odd? \nAND\n Is the number in the first half? \nAND\n Is then number prime?':[1,1,1,0,0],
                    'not odd':[0,0,0,1,0],
                    'not firsthalf':[0,0,0,0,1],
                    '-odd':[-1,0,0,0,0],
                    '-firsthalf':[0,-1,0,0,0],
                    '-prime':[0,0,-1,0,0],
                    'not odd-odd+prime':[-1,0,1,1,0],
                    'not odd+odd+prime':[1,0,1,1,0],
                    'not firsthalf+ firsthalf+prime':[0,1,1,0,1],
                    '-not odd+odd+prime':[1,0,1,-1,0],
                    'not odd+odd-prime':[1,0,-1,1,0]}

for text in test_rule_inputs:

    formatted_imgs=[]

    for img in val_imgs:

            # Build input by concatenating flattened image and rule input
            flattened_img=img.reshape(784)/255.
            rule_input=np.array(test_rule_inputs[text])
            final_input=np.concatenate([flattened_img, rule_input])
            formatted_imgs.append(final_input)

    formatted_imgs=np.array(formatted_imgs)

    out = new_model.predict(formatted_imgs)
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
    plt.savefig(f"{''.join([str(i) for i in test_rule_inputs[text]])}result.png")
    plt.clf()

