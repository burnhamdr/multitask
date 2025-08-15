import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run_sim():
    all_tasks=['firsthalf','notfirsthalf','under3','notunder3','under7','notunder7']
    train_tasks=['notunder7']
    learning_rate=0.001

    # Load data
    train_set,_=datasets.mnist.load_data()
    train_imgs,train_labels=train_set

    # Split into training and validation sets
    validation_split = 0.2
    split_index = int(len(train_imgs) * (1 - validation_split))
    train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
    train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

    train_imgs,train_labels=preprocessing(imgs=train_imgs,labels=train_labels,
                                        train_tasks=train_tasks,all_tasks=all_tasks)

    model = models.Sequential([
        layers.InputLayer(input_shape=(784+len(all_tasks),)),

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

    model_dir='./../../models/mlp_models/firsthalf_notfirsthalf_under3_notunder3_under7_10-32_001'
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

    num_epochs=10000

    performance_history=[]

    for i in range(num_epochs):

        ind=np.random.choice(len(train_imgs))
        img=train_imgs[ind,:]
        lab=train_labels[ind]
        img = img.reshape(1, -1)
        lab = np.array([lab])

        # Train the model
        train_history = model.fit(img, lab,
                            epochs=1)

        val_history={}

        for task in train_tasks:

            curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                                train_tasks=[task],all_tasks=all_tasks)

            val_history[task]=model.evaluate(curr_val_imgs,curr_val_labels)

        print(f'Epoch {i+1}/{num_epochs}')
        print(f'General training accuracy: \t',train_history.history['acc'][0])
        for task in train_tasks:
            print(f'{task} validation accuracy: \t', val_history[task][1])
            performance_history.append(val_history[task][1])

        if val_history['notunder7'][1] > 0.95:
            print("###############################################")
            break

    with open('./perf_profiles_pretrained_notunder7.txt','a') as file:
        
        file.write(','.join([str(x) for x in performance_history]))
        file.write('\n')
        print(performance_history)
    return

if __name__ == '__main__':
    run_sim()
