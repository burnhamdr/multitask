import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']
num_epochs=10
batch_size=64
learning_rate=0.005
num_neurons=32
train_tasks=['odd','firsthalf','prime','notodd','notfirsthalf']
model_dir='./../../models/mlp_models/odd_notodd_firsthalf_notfirsthalf_prime_full_10-32_005'

# Load data
train_set,_=datasets.mnist.load_data()
train_imgs,train_labels=train_set

# Split into training and validation sets
validation_split = 0.2
split_index = int(len(train_imgs) * (1 - validation_split))
train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

model = models.Sequential([
    layers.InputLayer(input_shape=(784+len(all_tasks),)),

    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
    layers.Dense(num_neurons, activation='relu'),
                                        
    layers.Dense(1, activation='sigmoid'),
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
    plt.savefig(f"{''.join([str(i) for i in test_rule_inputs[text]])}result.png")
    plt.clf()