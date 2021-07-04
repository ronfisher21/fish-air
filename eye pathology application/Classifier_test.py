from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import os
import efficientnet.keras as efn
from keras import Model, layers
from keras.models import load_model, model_from_json
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
import xlsxwriter
from openpyxl import load_workbook
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cv2
def cataract_classifier(img, filename, test_dir, final_model, train_dir, xlsx_results_loc, date_time=False): #changed!
    #final_model = load_model(classifier_mdl_dir)
    ######################## check ###################################
    img = np.array(img)
    img = cv2.resize(img, dsize=(600,600), interpolation=cv2.INTER_CUBIC)
    img = np.resize(img, (1, 600, 600, 3))
    pred = final_model.predict(img/255,verbose=1)
    predicted_class_indices = np.round(pred)
    labels={0:'Cataract', 1:'Healthy'}
    predictions = [labels[k[0]] for k in predicted_class_indices]
    #####################################################################
    xml_file_loc = xlsx_results_loc
    wb = load_workbook(xml_file_loc)
    ws = wb.worksheets[0]
    #added and modified!
    #for i in range(len(filenames)): ##### delete
    if date_time:
        ws.append([filename, predictions[0], datetime.now()]) #### modified
    else:
        ws.append([filename, predictions[0]]) ####modified
    wb.save(xml_file_loc)















