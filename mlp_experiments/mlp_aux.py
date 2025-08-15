import numpy as np
import tensorflow.keras.datasets as tfdatasets
from datasets import load_dataset
import os

def get_real_label(original_label,task):

    if task=='odd':
        new_label=original_label%2!=0
    if task=='firsthalf':
        new_label=original_label<5
    if task=='prime':
        new_label=1 if original_label in [2,3,5,7] else 0
    if task=='notodd':
        new_label=not get_real_label(original_label,'odd')
    if task=='notfirsthalf':
        new_label=not get_real_label(original_label,'firsthalf')
    if task=='notprime':
        new_label=not get_real_label(original_label,'prime')
    if task=='under3':
        new_label=original_label<3
    if task=='notunder3':
        new_label=original_label>=3
    if task=='under7':
        new_label=original_label<7
    if task=='notunder7':
        new_label=original_label>7
    if task=='under6':
        new_label=original_label<6

    return new_label

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
        dataset_dir=f'../../datasets/{dataset}'
        print(os.path.join(os.getcwd(),os.path.join(dataset_dir,f'{dataset}_MNIST_X_train.npy')))
        train_imgs=np.load(os.path.join(dataset_dir,f'{dataset}_MNIST_X_train.npy'),allow_pickle=True)
        train_labels=np.load(os.path.join(dataset_dir,f'{dataset}_MNIST_y_train.npy'),allow_pickle=True)
        train_set=train_imgs,train_labels
    else:
        raise Exception('unsupported dataset')

    train_imgs,train_labels=train_set
    print(train_imgs.shape)
    print(train_labels.shape)

    # Split into training and validation sets
    validation_split = 0.2
    split_index = int(len(train_imgs) * (1 - validation_split))
    train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
    train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

    # Preprocess data
    train_imgs,train_labels=preprocessing(imgs=train_imgs,labels=train_labels,
                                        train_tasks=[],all_tasks=[],class_training=True, no_rule=True)

    return train_imgs,train_labels, val_imgs,val_labels