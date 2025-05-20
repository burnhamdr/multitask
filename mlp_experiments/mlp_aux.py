import numpy as np

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

    return new_label

def preprocessing(imgs,labels,train_tasks,all_tasks,prob_tasks=None, class_training=False):

    if class_training:

        formatted_imgs=[]
        formatted_labels=[]

        for img,label in zip(imgs,labels):
            
            flattened_img=img.reshape(784)/255.
            rule_input=np.zeros(len(all_tasks))
            final_input=np.concatenate([flattened_img, rule_input])
            formatted_imgs.append(final_input)

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
