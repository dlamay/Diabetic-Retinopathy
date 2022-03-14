import tensorflow
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow import keras
import glob
import os

from read_images import read_dataset

from ResNet50 import resnet50
from tensorflow.keras.applications.vgg19 import VGG19

cnn_name = 'resnet'

# paths
log_path = r'/scratch/lamay.d/training_log'
checkpoint_path = r'/scratch/lamay.d/training_checkpoints'
#date = '0312'
#number_of_training_on_date = '1'
#file_name = 'autoencoder_'+date+'_'+number_of_training_on_date
#log_path = os.path.join(log_root_path, file_name)
#checkpoint_path = os.path.join(checkpoint_root_path, file_name)

# define parameters
val_ratio = 0.2
batch_size = 64
target_size = (400, 600)
input_size = (400, 600, 3)
# (500, 750)

#set up datasets for training
train_ds, val_ds = read_dataset(
    model_type = 'cnn',
    batch_size = batch_size,
    split_ratio = val_ratio,
    target_size = target_size)



# for img, label in train:
#   print(img.shape)
#   print(label.shape)
#   break

# set up auto encoder model
# try: # find the latest training checkpoint
#     list_of_files = glob.glob(os.path.join(checkpoint_path, '*'))
#     latest_file = max(list_of_files, key = os.path.getctime) # if training checkpoint exists
#     autoencoder = keras.models.load_model(latest_file) # load the checkpoint
# except: # if there is no checkpoint found
#     autoencoder = AUTOENCODER(input_size) # build the model from start

if cnn_name == 'resnet':
    # set up auto encoder model
    #if os.path.isdir(log_path) == False: # if there is no folder for current training
        #print('No folder found for this training, creating folder')
        # create folders for this training
        #os.makedirs(log_path)
        #os.makedirs(checkpoint_path)
        # start new model
        #cnn = resnet50(input_size)
        #for layer in cnn.layers:
           # layer.trainable = False
        #flattened_final_weights = Flatten()(cnn.output)
        #output = Dense(5,activation='softmax')(flattened_final_weights)
        #cnn = Model(inputs=cnn.inputs, outputs=output)
    #else: # if folder is found for current training
        #print('Folder found for this training, looking for latest checkpoint')
        #try: # try to find latested checkpoint record
            #list_of_files = glob.glob(os.path.join(checkpoint_path, '*')) #go through existing list of checkpoints
            #latest_file = max(list_of_files, key = os.path.getctime) # if training checkpoint exists
            #print('Latest checkpoint found, continue training')
            #cnn = keras.models.load_model(latest_file) # load the checkpoint
            #for layer in cnn.layers:
                #layer.trainable = False
            #flattened_final_weights = Flatten()(cnn.output)
            #output = Dense(5,activation='softmax')(flattened_final_weights)
            #cnn = Model(inputs=cnn.inputs, outputs=output)
        #except: #if no files found, i.e. existing folder is empty
            #print('No checkpoint found, start new training')
            #cnn = resnet50(input_size) # start new model
    
    cnn = resnet50(input_size)  
    for layer in cnn.layers:
        layer.trainable = False
    #flattened_final_weights = Flatten()(cnn.output)
    #output = Dense(5,activation='softmax')(flattened_final_weights)
    #cnn = Model(inputs=cnn.inputs, outputs=output)
    
    
elif cnn_name == 'vgg':
    cnn = VGG19(
        weights = 'imagenet',
        include_top = False,
        classes = 5)

# set up callback functions
cnn_callbacks = [
    tensorflow.keras.callbacks.CSVLogger(
        filename = os.path.join(log_path, cnn_name + '_noencoder__0305.log'),
        append = True # append if file exists, i.e. for continuing training
    ),
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_path, cnn_name + '_noencoder_0305_{epoch:02d}.hdf5'),
        save_freq = 'epoch'
    ),
    # keras.callbacks.TerminateOnNan(),
    tensorflow.keras.callbacks.ReduceLROnPlateau(
        patience = 3
    )
]

# complile the model
# focal loss function reference: https://focal-loss.readthedocs.io/en/latest/generated/focal_loss.SparseCategoricalFocalLoss.html
cnn.compile(
    optimizer="Adam",
    # loss=keras.losses.CategoricalCrossentropy(),
    loss = tensorflow.keras.losses.CategoricalCrossentropy(),
    metrics = [tensorflow.keras.metrics.Precision(),
               tensorflow.keras.metrics.Recall(),
               tensorflow.keras.metrics.AUC()]
)

# train and validate the model
cnn.fit(train_ds,
        epochs=2,
        batch_size=batch_size,
        shuffle=True,
        validation_data=val_ds,
        callbacks = cnn_callbacks
)

# def main():    # you can name this whatever you want, it doesn't need to be main()
#     train_cnn()
#
# if __name__ == '__main__':
#     main()