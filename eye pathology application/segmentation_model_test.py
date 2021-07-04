# load and evaluate a saved model
from numpy import loadtxt
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2






def Image_Cropper(img_path, cropped_img_path , cropped_mask_path, h5_file_location):
    #Support functions
    def read_image(path, IMAGE_SIZE=512):
        img_name = path.split("\\")[-1]
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0
        return x, img_name

    def model():
        IMAGE_SIZE = 512
        inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

        encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=1.4)
        skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
        encoder_output = encoder.get_layer("block_13_expand_relu").output

        f = [16, 32, 48, 64]  # add features
        x = encoder_output
        for i in range(1, len(skip_connection_names) + 1, 1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2, 2))(x)
            x = Concatenate()([x, x_skip])

            x = Conv2D(f[-i], (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(f[-i], (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        x = Conv2D(1, (1, 1), padding="same")(x)
        x = Activation("sigmoid")(x)

        model = Model(inputs, x)
        return model



    #Main func

    # load model
    model = model()
    model.load_weights(h5_file_location)
    # save  prediction
    img, img_name = read_image(img_path)
    #added!!!!
    if img_name.split(".")[-1]!='jpg' or'JPG':
        img_name_split = img_name.split(".")
        img_name = img_name_split[:-1][0] + '.jpg'
    ###########3
    pred = model.predict(np.expand_dims(img, axis=0))[0] > 0.5
    pred_img = pred.astype(np.uint8)
    pred_img_mask = pred_img * 255
    cropped = pred_img * img
    plt.imsave(os.path.join(cropped_img_path, img_name), cropped)
    mask_name = "mask_" + img_name
    plt.imsave(os.path.join(cropped_mask_path, mask_name), pred_img_mask.squeeze())
    return cropped, pred_img_mask.squeeze(), img_name









