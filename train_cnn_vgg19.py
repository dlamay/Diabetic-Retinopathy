import glob
import os
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from read_images import read_dataset

#from ResNet50 import resnet50
from tensorflow.keras.applications.vgg19 import VGG19

cnn_name = 'vgg'

# paths
log_path = r'/scratch/anchawale.m/training_log'
checkpoint_path = r'/scratch/anchawale.m/training_checkpoints'

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

# if cnn_name == 'resnet':
#     cnn = resnet50(input_size)
if cnn_name == 'vgg':
    cnn = VGG19(
        weights = 'imagenet',
        input_shape=(400, 600, 3),
        include_top = False)
        #classes = 5)
    for layer in cnn.layers:
        layer.trainable = False

    flattened_final_weights = Flatten()(cnn.output)

    case_specific_training_layer1 = Dense(100, activation='relu')(flattened_final_weights)
    regularized_case_specific_training_1 = Dropout(0.2, seed=67)(case_specific_training_layer1)
    case_specific_training_layer2 = Dense(100, activation='relu')(regularized_case_specific_training_1)
    regularized_case_specific_training_2 = Dropout(0.2, seed=66)(case_specific_training_layer2)
    case_specific_training_layer3 = Dense(100, activation='relu')(regularized_case_specific_training_2)
    regularized_case_specific_training_3 = Dropout(0.2, seed=65)(case_specific_training_layer3)

    vgg_output = Dense(5, activation='softmax')(regularized_case_specific_training_3)
    cnn = Model(inputs=cnn.inputs, outputs=vgg_output)
    plot_model(cnn, to_file='/scratch/anchawale.m/Figures/vgg_model_plot.png', show_shapes=True, show_layer_names=True)

# set up callback functions
cnn_callbacks = [
    tensorflow.keras.callbacks.CSVLogger(
        filename = os.path.join(log_path, cnn_name + '_noencoder__0312_1.log'),
        append = True # append if file exists, i.e. for continuing training
    ),
    tensorflow.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_path, cnn_name + '_noencoder_0312_1_{epoch:02d}.hdf5'),
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
    #loss=tensorflow.keras.losses.CategoricalCrossentropy(),
    #loss = keras.losses.SparseCategoricalCrossentropy(),
    loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
    metrics = [tensorflow.keras.metrics.Precision(),
               tensorflow.keras.metrics.Recall(),
               tensorflow.keras.metrics.AUC()]
)

# train and validate the model
fitted_model = cnn.fit(train_ds,
        epochs=10,
        batch_size=batch_size,
        shuffle=True,
        validation_data=val_ds,
        callbacks = cnn_callbacks)

# Plot and save accuracy vs epoch
plt.plot(fitted_model.history['accuracy'])
plt.plot(fitted_model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.show()
plt.savefig("/scratch/anchawale.m/Figures/acc_vs_epoch_vgg_1.png")

# Plot and save loss vs epoch
plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.show()
plt.savefig("/scratch/anchawale.m/Figures/loss_vs_epoch_vgg_1.png")

# def main():    # you can name this whatever you want, it doesn't need to be main()
#     train_cnn()
#
# if __name__ == '__main__':
#     main()