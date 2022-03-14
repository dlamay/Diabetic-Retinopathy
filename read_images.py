import os
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

def read_dataset(model_type, train = True, batch_size = 64, split_ratio = 0.15, target_size = (1000, 1500)):
    # file paths
    image_train_path = r'/scratch/wang.tads\train\train\trainLabels.csv'
    image_test_path = r'C:\Users\tonia\Downloads\test\test'
    label_path = r'/scratch/wang.tongx/ds5500/images/trainLabels.csv'
    # image_test_path = r'/scratch/wang.tongx/ds5500/images/test'
    #     # image_train_path = r'C:\Users\tonia\Downloads\train\train\train'
    #     # label_path = r'C:\Users\tonia\Downloads\test\test'

    if train:
        # load image information into dataframe
        train_labels_df = pd.read_csv(label_path)
        train_labels_df['level'] = train_labels_df['level'].astype(str)
        train_labels_df['image'] = train_labels_df.apply(lambda row: row['image']+'.jpeg', axis=1)
        if model_type == 'autoencoder':
            # data generator for autoencoder
            autoencoder_datagen = ImageDataGenerator(
                validation_split=split_ratio,
                samplewise_std_normalization=True,
                rotation_range=5,  # degree range for random rotations
                brightness_range=(0.5, 1.5),
                zoom_range=0.1,  # zoom range = [0.9, 1.1]
                horizontal_flip=True)  # splitting 15% to be validation data

            autoencoder_train_generator = autoencoder_datagen.flow_from_dataframe(
                dataframe=train_labels_df,
                directory=image_train_path,
                x_col="image",
                target_size=target_size,  # resizing images to 1000x1500
                batch_size=batch_size,  # batch size to be determined
                class_mode="input",
                subset='training',
                shuffle=True,
                seed=42
            )

            autoencoder_val_generator = autoencoder_datagen.flow_from_dataframe(
                dataframe=train_labels_df,
                directory=image_train_path,
                x_col="image",
                target_size=target_size,  # resizing images to 1000x1500
                batch_size=batch_size,  # batch size to be determined
                class_mode="input",
                subset='validation',
                shuffle=True,
                seed=42
            )

            return autoencoder_train_generator, autoencoder_val_generator

        elif model_type == 'cnn':
            cnn_datagen = ImageDataGenerator(
                validation_split=split_ratio,
                samplewise_std_normalization=True,
                rotation_range=5,  # degree range for random rotations
                brightness_range=(0.5, 1.5),
                zoom_range=0.1,  # zoom range = [0.9, 1.1]
                horizontal_flip=True)  # splitting 15% to be validation data

            cnn_train_generator = cnn_datagen.flow_from_dataframe(
                dataframe=train_labels_df,
                directory=image_train_path,
                x_col="image",
                y_col='level',
                target_size=target_size,  # resizing images to 1000x1500
                batch_size=batch_size,  # batch size to be determined
                class_mode="categorical",
                subset='training',
                shuffle=True,
                seed=42
            )

            cnn_val_generator = cnn_datagen.flow_from_dataframe(
                dataframe=train_labels_df,
                directory=image_train_path,
                x_col="image",
                y_col='level',
                target_size=target_size,  # resizing images to 1000x1500
                batch_size=batch_size,  # batch size to be determined
                class_mode='categorical',
                subset='validation',
                shuffle=True,
                seed=42
            )

            return cnn_train_generator, cnn_val_generator

    else:
        test_filenames = os.listdir(image_test_path)
        test_df = pd.DataFrame(test_filenames, columns = ['image'])
        if model_type == 'autoencoder':
            autoencoder_test_datagen = ImageDataGenerator(
                samplewise_std_normalization=True)

            autoencoder_test_generator = autoencoder_test_datagen.flow_from_dataframe(
                dataframe=test_df,
                directory=image_test_path,
                x_col="image",
                y_col = None,
                class_mode = None,
                target_size = target_size,
                batch_size = batch_size,
                shuffle = True,
                seed = 42
            )
            return autoencoder_test_generator

        elif model_type == 'cnn':
            cnn_test_datagen = ImageDataGenerator(
                samplewise_std_normalization=True)

            cnn_test_generator = cnn_test_datagen.flow_from_dataframe(
                dataframe=test_df,
                directory=image_test_path,
                x_col="image",
                y_col='level',
                class_mode='categorical',
                target_size=target_size,
                batch_size=batch_size,
                shuffle=True,
                seed=42
            )
        return cnn_test_generator