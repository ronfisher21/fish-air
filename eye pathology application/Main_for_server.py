from segmentation_model_test import *
from segmentation_result_morphological_func import *
from Classifier_test import *

main_dir_cropper = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\cropper'
main_dir_classifier = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\classifier'
save_cropped_to = r"C:\Users\ron\Desktop\auto crop and prediction check\mine\Cropped\Cataract" #added!
#Step 1 - Centered 512x512 picture Retrieval
#Dan Insert code for picture retrieval here


#Step 2 - automatic segmentation
#TODO: Discuss with Dan about the prefered cropped im and mask locations
Centered_im_path_main = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\val'

Cropped_pred_im_path = os.path.join(main_dir_cropper,'res')
Cropped_pred_mask_path = os.path.join(main_dir_cropper,'res')
weights_dir = os.path.join(main_dir_cropper,'model_1_30_epochs_alpha1.4_balanced_augmented.h5')
images_names = os.listdir(Centered_im_path_main)
#new!
binary_mask_dict = {}
cropped_image_dict = {}
edges_dict = {}
flag_dict = {}

#First we are looping on all images and cropping the images only and saving them in the save_cropped_to directory
for image_name in images_names:
    Centered_im_path = os.path.join(Centered_im_path_main,image_name)
    cropped_img, img_mask, img_name = Image_Cropper(Centered_im_path, Cropped_pred_im_path, Cropped_pred_mask_path, weights_dir) #if was added in Image_Cropper

#Step 3 - Additional image processing
    target_dir = os.path.join(main_dir_cropper,'res_after_morpho')
    binary_mask, cropped_image, edges, flag = segmentation_post_proccess(cropped_img, img_mask, target_dir, img_name)
    binary_mask_dict[img_name] = binary_mask #added!
    cropped_image_dict[img_name] = cropped_image #added!
    edges_dict[img_name] = edges #added!
    flag_dict[img_name] = flag #added!
    image_full_path = os.path.join(save_cropped_to, img_name) #added!
    cropped_image = cropped_image.save(image_full_path) #added! the directory save cropped_to needs to contain 2 folder, Cataract and Healthy
#Step 4 - image with dashed borders lines
#Dan Use edges and step 1 image

######## This step will be outside the for loop, we first looping on all val images, copping them and saving on different folder called Cropped
#Step 5 - Classification
classifier_mdl_dir = os.path.join(main_dir_classifier,'model_effecientnet_generator4_with_ADAM_batch_64_50_epochs_Augmented_data_28.3.h5')
train_dir = os.path.join(main_dir_classifier,'train')
test_dir = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\Cropped' #directory of the cropped images that will be the 1st input of cataract_classifier. in this directory the data should be divided to 2 folders: Healthy and Cataract
csv_result_file = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\results_new.xlsx'

###################check##########################
folder_dest = r'C:\Users\ron\Desktop\auto crop and prediction check\mine\run_without_test_generator\images\talget\Cropped_Img'
final_model = load_model(classifier_mdl_dir)
#for check_name in os.listdir(folder_dest):
img = Image.open(os.path.join(folder_dest,check_name))
filename = check_name
predictions = cataract_classifier(img, filename, test_dir, final_model, train_dir, csv_result_file, date_time=True) #the second input, cropped_image is removed and instead the first input is the directory of all cropped images
#image name is also removed
#Date time True will generate date and time of the process to the final excel
