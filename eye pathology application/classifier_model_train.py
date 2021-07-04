import tensorflow as tf
import keras
import PIL
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
#from keras.applications.efficientnet import Efficientnet
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import efficientnet.keras as efn

#Data partitioning
train_dir =r"/content/drive/My Drive/app/cataract/data/classification_after_augmentation/train/"
val_dir =r"/content/drive/My Drive/app/cataract/data/classification_after_augmentation/val/"
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator_4 = train_datagen.flow_from_directory(train_dir, target_size=(600, 600), batch_size=64, class_mode='binary')
validation_generator_4 = test_datagen.flow_from_directory(val_dir, target_size=(600, 600), batch_size=96, class_mode='binary')


conv_base = efn.EfficientNetB7(include_top=False, input_shape=(600, 600, 3), weights='noisy-student') # original weights were imagenet

for layer in conv_base.layers:
    layer.trainable = False

#Trainable part
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)


############################################ TEST 1 - ADAM optimizer ################################################

optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


#Training

history = model.fit_generator(generator=train_generator_4,
                              epochs=50,
                              validation_data=validation_generator_4)

#Saving the model

# architecture and weights to HDF5
save_mdl_dir = r'/content/drive/My Drive/app/cataract/models/model_effecientnet_generator4_with_ADAM_batch_64_50_epochs_additional_Data_and_augmentation_600.h5'
model.save(save_mdl_dir)


##################################### TEST 2 - SGD optimizer ##############################################

optimizer = keras.optimizers.SGD()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


#Training

history = model.fit_generator(generator=train_generator_4,
                              epochs=50,
                              validation_data=validation_generator_4)

#Saving the model

# architecture and weights to HDF5
save_mdl_dir = r'/content/drive/My Drive/app/cataract/models/model_effecientnet_generator4_with_SGD_batch_64_50_epochs_additional_Data_and_augmentation_600.h5'
model.save(save_mdl_dir)