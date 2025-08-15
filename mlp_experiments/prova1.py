import os
import numpy as np
from mlp_aux import get_data,preprocessing
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_dir='./../../models/mlp_models/paper_unif'

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

weights=np.load(os.path.join(model_dir,'weights.npy'),allow_pickle=True)
in_weights=weights[0]
out_weights=weights[1]

bias_vai=np.load(os.path.join(model_dir,'Vai_biases.npy'),allow_pickle=True)
bias_osmanya=np.load(os.path.join(model_dir,'Osmanya_biases.npy'),allow_pickle=True)
bias_nko=np.load(os.path.join(model_dir,'NKo_biases.npy'),allow_pickle=True)

vai_data=np.zeros((11,11,11))
osmanya_data=np.zeros((11,11,11))
nko_data=np.zeros((11,11,11))

train_imgs_vai,train_labels_vai,val_imgs_vai,val_labels_vai=get_data('Vai')
curr_val_imgs_vai,curr_val_labels_vai=preprocessing(imgs=val_imgs_vai,labels=val_labels_vai,
                                                train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

train_imgs_osm,train_labels_osm,val_imgs_osm,val_labels_osm=get_data('Osmanya')
curr_val_imgs_osm,curr_val_labels_osm=preprocessing(imgs=val_imgs_osm,labels=val_labels_osm,
                                                train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

train_imgs_nko,train_labels_nko,val_imgs_nko,val_labels_nko=get_data('NKo')
curr_val_imgs_nko,curr_val_labels_nko=preprocessing(imgs=val_imgs_nko,labels=val_labels_nko,
                                                train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

for i,a in enumerate(np.linspace(0,1,11)):
    for j,b in enumerate(np.linspace(0,1,11)):
        for k,c in enumerate(np.linspace(0,1,11)):
            in_biases=a*bias_vai[0]+b*bias_osmanya[0]+c*bias_nko[0]
            out_biases=a*bias_vai[1]+b*bias_osmanya[1]+c*bias_nko[1]

            model = models.Sequential([
            layers.InputLayer(input_shape=(784,)),
            layers.Dense(30000, use_bias=False, trainable=False, kernel_initializer='uniform'),
            BiasOnlyLayer(30000, activation='relu'),
            layers.Dense(10, use_bias=False, trainable=False, kernel_initializer='uniform'),
            BiasOnlyLayer(10, activation='softmax'),
            ])

            model.layers[0].set_weights([in_weights]) 
            model.layers[1].set_weights([in_biases])  
            model.layers[2].set_weights([out_weights])
            model.layers[3].set_weights([out_biases])

            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            print(f"a: {a}, b: {b}, c:{c}")

            _,acc=model.evaluate(curr_val_imgs_vai,curr_val_labels_vai)
            vai_data[i,j,k]=acc

            _,acc=model.evaluate(curr_val_imgs_osm,curr_val_labels_osm)
            osmanya_data[i,j,k]=acc

            _,acc=model.evaluate(curr_val_imgs_nko,curr_val_labels_nko)
            nko_data[i,j,k]=acc

np.save(os.path.join(model_dir,'vai_lin_comp_perfs3_retry.npy'),vai_data)
np.save(os.path.join(model_dir,'osmanya_lin_comp_perfs3_retry.npy'),osmanya_data)
np.save(os.path.join(model_dir,'nko_lin_comp_perfs3_retry.npy'),nko_data)