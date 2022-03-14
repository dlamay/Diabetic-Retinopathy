import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import glob
import os

from AutoEncoder import AUTOENCODER
from read_images import read_dataset

# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg19 import VGG19

# paths
log_root_path = r'/scratch/wang.tongx/ds5500/training_log'
checkpoint_root_path = r'/scratch/wang.tongx/ds5500/training_checkpoints'
date = '0309'
number_of_training_on_date = '2'
file_name = 'autoencoder_'+date+'_'+number_of_training_on_date
log_path = os.path.join(log_root_path, file_name)
checkpoint_path = os.path.join(checkpoint_root_path, file_name)

# define parameters
val_ratio = 0.2
batch_size = 128
target_size = (400, 600)
input_size = (400, 600, 3)
# (500, 750)

#set up datasets for training
train_ds, val_ds = read_dataset(
    model_type = 'autoencoder',
    batch_size = batch_size,
    split_ratio = val_ratio,
    target_size = target_size)


# set up auto encoder model
if os.path.isdir(log_path) == False: # if there is no folder for current training
    print('No folder found for this training, creating folder')
    # create folders for this training
    os.makedirs(log_path)
    os.makedirs(checkpoint_path)
    # start new model
    autoencoder = AUTOENCODER(input_size)
else: # if folder is found for current training
    print('Folder found for this training, looking for latest checkpoint')
    try: # try to find latested checkpoint record
        list_of_files = glob.glob(os.path.join(checkpoint_path, '*')) #go through existing list of checkpoints
        latest_file = max(list_of_files, key = os.path.getctime) # if training checkpoint exists
        autoencoder = keras.models.load_model(latest_file) # load the checkpoint
        print('Latest checkpoint found, continue training')
    except: #if no files found, i.e. existing folder is empty
        print('No checkpoint found, start new training')
        autoencoder = AUTOENCODER(input_size) # start new model

# set up callback functions
autoencoder_callbacks = [
    keras.callbacks.CSVLogger(
        filename = os.path.join(log_path, file_name+'.log'),
        append = True # append if file exists, i.e. for continuing training
    ),
    keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_path, file_name+'_h{epoch:02d}.df5'),
        save_freq = 'epoch'
    ),
    # keras.callbacks.TerminateOnNan(),
    keras.callbacks.ReduceLROnPlateau(
        patience = 5
    )
]

# complile the model
autoencoder.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.01),
    loss="mean_squared_error"
)

# train and validate the model
autoencoder.fit(train_ds,
                epochs=50,
                batch_size=batch_size,
                shuffle=True,
                validation_data=val_ds,
                callbacks = autoencoder_callbacks
                )
