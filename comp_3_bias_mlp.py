import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as tfdatasets
from datasets import load_dataset
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

init=tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1)

def preprocessing(imgs,labels,train_tasks,all_tasks,prob_tasks=None, class_training=False, no_rule=False):

    if class_training:

        formatted_imgs=[]
        formatted_labels=[]

        for img,label in zip(imgs,labels):
            
            flattened_img=img.reshape(784)/255.
            if not no_rule:
                rule_input=np.zeros(len(all_tasks))
                flattened_img=np.concatenate([flattened_img, rule_input])
            formatted_imgs.append(flattened_img)

            # Redefine the corresponding label based on which task we have extracted
            
            formatted_labels.append(label)


        formatted_imgs=np.array(formatted_imgs)
        formatted_labels=np.array(formatted_labels)

        return formatted_imgs,formatted_labels

    if prob_tasks==None or len(prob_tasks)!=len(train_tasks):
        prob_tasks=[1./len(train_tasks)]*len(train_tasks)

    formatted_imgs=[]
    formatted_labels=[]

    for img,label in zip(imgs,labels):
        
        # Define task according to probability distribution
        curr_task=np.random.choice(train_tasks,p=prob_tasks)

        # Build input by concatenating flattened image and rule input
        flattened_img=img.reshape(784)/255.
        rule_input=np.zeros(len(all_tasks))
        rule_input[all_tasks.index(curr_task)]=1
        final_input=np.concatenate([flattened_img, rule_input])
        formatted_imgs.append(final_input)

        # Redefine the corresponding label based on which task we have extracted
        
        formatted_labels.append(get_real_label(label,curr_task))


    formatted_imgs=np.array(formatted_imgs)
    formatted_labels=np.array(formatted_labels)

    return formatted_imgs,formatted_labels

def get_data(dataset):

    # Load data
    if dataset == 'Mnist':
        train_set,_=tfdatasets.mnist.load_data()
    elif dataset == 'Fashion':
        train_set,_=tfdatasets.fashion_mnist.load_data()
    elif dataset == 'Kmnist':
        dataset = load_dataset("tanganke/kmnist")
        train_set = (np.array([np.array(img) for img in dataset['train']['image']]), np.array(dataset['train']['label']))
    elif dataset in ['Ethiopic', 'NKo', 'Osmanya', 'Vai']:
        dataset_dir=f'../datasets/{dataset}'
        train_imgs=np.load(os.path.join(dataset_dir,f'{dataset}_MNIST_X_train.npy'),allow_pickle=True)
        train_labels=np.load(os.path.join(dataset_dir,f'{dataset}_MNIST_y_train.npy'),allow_pickle=True)
        train_set=train_imgs,train_labels
    else:
        raise Exception('unsupported dataset')

    train_imgs,train_labels=train_set

    # Split into training and validation sets
    validation_split = 0.2
    split_index = int(len(train_imgs) * (1 - validation_split))
    train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
    train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

    # Preprocess data
    train_imgs,train_labels=preprocessing(imgs=train_imgs,labels=train_labels,
                                        train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

    return train_imgs,train_labels, val_imgs,val_labels
model_dir='./../models/mlp_models/paper_unif'

weights=np.load(os.path.join(model_dir,'weights.npy'),allow_pickle=True)
in_weights=weights[0]
out_weights=weights[1]

bias_vai=np.load(os.path.join(model_dir,'Vai_biases.npy'),allow_pickle=True)
bias_osmanya=np.load(os.path.join(model_dir,'Osmanya_biases.npy'),allow_pickle=True)
bias_nko=np.load(os.path.join(model_dir,'NKo_biases.npy'),allow_pickle=True)

vai_data=np.zeros((11,11,11))
osmanya_data=np.zeros((11,11,11))
nko_data=np.zeros((11,11,11))

for i,a in enumerate(np.linspace(0,1,11)):
    for j,b in enumerate(np.linspace(0,1,11)):
        for k,c in enumerate(np.linspace(0,1,11)):

            tf.keras.backend.clear_session()

            model = models.Sequential([
            layers.InputLayer(input_shape=(784,)),
            layers.Dense(30000, use_bias=False, trainable=False, kernel_initializer='uniform'),
            BiasOnlyLayer(30000, activation='relu'),
            layers.Dense(10, use_bias=False, trainable=False, kernel_initializer='uniform'),
            BiasOnlyLayer(10, activation='softmax'),
            ])

            in_biases=a*bias_vai[0]+b*bias_osmanya[0]+c*bias_nko[0]
            out_biases=a*bias_vai[1]+b*bias_osmanya[1]+c*bias_nko[1]

            model.layers[0].set_weights([in_weights]) 
            model.layers[1].set_weights([in_biases])  
            model.layers[2].set_weights([out_weights])
            model.layers[3].set_weights([out_biases])

            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            print(f"a: {a}, b: {b}, c:{c}")

            train_imgs,train_labels,val_imgs,val_labels=get_data('Vai')
            curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                                            train_tasks=[],all_tasks=[],class_training=True, no_rule=True)
            _,acc=model.evaluate(curr_val_imgs,curr_val_labels)
            vai_data[i,j,k]=acc
            
            train_imgs,train_labels,val_imgs,val_labels=get_data('Osmanya')
            curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                                            train_tasks=[],all_tasks=[],class_training=True, no_rule=True)
            _,acc=model.evaluate(curr_val_imgs,curr_val_labels)
            osmanya_data[i,j,k]=acc

            train_imgs,train_labels,val_imgs,val_labels=get_data('NKo')
            curr_val_imgs,curr_val_labels=preprocessing(imgs=val_imgs,labels=val_labels,
                                                            train_tasks=[],all_tasks=[],class_training=True, no_rule=True)
            _,acc=model.evaluate(curr_val_imgs,curr_val_labels)
            nko_data[i,j,k]=acc

np.save(os.path.join(model_dir,'3way','vai_lin_comp_perfs.npy'),vai_data)
np.save(os.path.join(model_dir,'3way','osmanya_lin_comp_perfs.npy'),osmanya_data)
np.save(os.path.join(model_dir,'3way','nko_lin_comp_perfs.npy'),nko_data)