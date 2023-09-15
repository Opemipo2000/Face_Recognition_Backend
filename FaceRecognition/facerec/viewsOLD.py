from django.shortcuts import render

# Create your views here.
import numpy as np
import pandas as pd
import cv2
import os
import pickle
from PIL import Image
# import dlib
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
from imutils import face_utils

# Remember to replace "from keras.engine.topology import get_source_inputs" to "from keras.utils.layer_utils import get_source_inputs" in the keras_vggface/models.py file.
from keras_vggface.vggface import VGGFace

import keras.utils as image
from keras_vggface import utils

import tensorflow.keras as keras

from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.engine.training import Model

from tensorflow.keras.models import load_model


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def image_preprocessor(new_image):
    # load the pretrained face detection by OpenCV
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # adjust the brightness of the image
    new_image = cv2.normalize(new_image, None, alpha=0, beta=1.5*255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert image to gray scale OpenCV
    gray_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Detect face using haar cascade classifier
    faces_coordinates = face_detector.detectMultiScale(gray_img)

    # Draw a rectangle around the face of the image and crop out the face to preprocess it.
    for (x, y, w, h) in faces_coordinates:
        face = new_image[y:y+h, x:x+w]
        face = cv2.resize(face, [224,224])
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = utils.preprocess_input(face, version=1)

        return face
    
def register_details(video,first_name,last_name):
    vidcap = cv2.VideoCapture(video)
    os.makedirs("./Headshots/{}_{}".format(first_name,last_name))
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("./Headshots/{}_{}/image".format(first_name,last_name)+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    
    sec = 0
    frameRate = 2 # capture image in each 2 seconds
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

    images_dir = "./Headshots"
    current_id = 0
    label_ids = {}
    for root, _, files in os.walk(images_dir):
        for file in files:
            if (file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg")):
                 # path of the image
                path = os.path.join(root, file)

                 # get the label name (name of the person)
                label = os.path.basename(root).replace("_", ".").lower()

                # add the label (key) and its number (value)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # load the image
                image_array = cv2.imread(path, cv2.IMREAD_COLOR)

                # Convert image to gray scale OpenCV
                gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

                # get the faces detected in the image
                faces_coordinates = face_detector.detectMultiScale(gray_img)

                # if not exactly 1 face is detected, skip this photo
                if len(faces_coordinates) != 1:
                    print(f'---Photo skipped---\n')
                    # remove the original image
                    os.remove(path)
                    continue

                # save the detected face(s) and associate
                # them with the label
                for (x, y, w, h) in faces_coordinates:
                    face = image_array[y:y+h, x:x+w]
                    face = cv2.resize(face, [224,224])
                    image_array = np.array(face, "uint8")

                    # remove the original image
                    os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    im.save(path)


    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(images_dir, target_size=(224,224), color_mode='rgb', batch_size=32, class_mode='categorical', shuffle=True)

    NO_CLASSES = len(train_generator.class_indices.values())    

    base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

    x = base_model.output

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(units=1024, kernel_initializer="lecun_normal", activation="selu")(x)
    x = keras.layers.Dense(units=1024, kernel_initializer="lecun_normal", activation="selu")(x)
    x = keras.layers.Dense(units=512, kernel_initializer="lecun_normal", activation="selu")(x)

    # final layer with softmax activation
    preds = keras.layers.Dense(units=NO_CLASSES, activation='softmax')(x)

    # create a new model with the base model's original input and the
    # new model's output
    model = Model(inputs = base_model.input, outputs = preds)

    # don't train the first 286 layers - 0..18
    for layer in model.layers[:19]:
        layer.trainable = False

    # train the rest of the layers - 19 onwards
    for layer in model.layers[19:]:
        layer.trainable = True

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

    model.fit(train_generator, batch_size = 1, verbose = 1, epochs = 5)

    # creates a HDF5 file
    model.save('./h5_folder/{}_{}_transfer_learning_trained'.format(first_name,last_name) + '_face_cnn_model.h5')

    # deletes the existing model
    del model

    # saving training labels
    class_dictionary = train_generator.class_indices
    class_dictionary = {
        value:key for key, value in class_dictionary.items()
    }

    # save the class dictionary to pickle
    face_label_filename = './labels_folder/{}_{}_face-labels.pickle'.format(first_name,last_name)
    with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)

    print("Registration complete")
    
def identify_face(image,first_name,last_name):
    # returns a compiled model identical to the previous one
    model = load_model('./h5_folder/{}_{}_transfer_learning_trained'.format(first_name,last_name) + '_face_cnn_model.h5')

    # load the training labels
    face_label_filename = './labels_folder/{}_{}_face-labels.pickle'.format(first_name,last_name)
    with open(face_label_filename, "rb") as \
        f: class_dictionary = pickle.load(f)

    class_list = [value for _, value in class_dictionary.items()]

    # prepare the image
    processed_test_img = image_preprocessor(image)
    # perform prediction
    test_preds = model.predict(processed_test_img)

    identity = class_list[test_preds[0].argmax()]

    return identity

def calculate_EAR(eye):
  
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
  
    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])
  
    # calculate the EAR
    EAR = (y1+y2) / x1
    return EAR

def calculate_face_attentiveness(video_path):
    # Variables
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
    blink_count = 0
    mean_blink_rate = 15
    std_blink_rate = 3
    
    # Eye landmarks
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    
    # Initializing the Models for Landmark and 
    # face Detection
    detector = dlib.get_frontal_face_detector()
    landmark_predict = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    cam = cv2.VideoCapture(video_path)

    # Gets duration of the video
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = frame_count / fps
    minutes = seconds / 60

    while True:
        _, frame = cam.read()

        if frame is None:
            break  # Exit the loop when the video ends

        frame = imutils.resize(frame, width=640)
  
        # converting frame to gray scale to
        # pass to detector
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
        # detecting the faces
        faces = detector(img_gray)
        for face in faces:
  
            # landmark detection
            shape = landmark_predict(img_gray, face)
  
            # converting the shape class directly
            # to a list of (x,y) coordinates
            shape = face_utils.shape_to_np(shape)
  
            # parsing the landmarks list to extract
            # lefteye and righteye landmarks--#
            lefteye = shape[L_start: L_end]
            righteye = shape[R_start:R_end]
  
            # Calculate the EAR
            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)
  
            # Avg of left and right eye EAR
            avg = (left_EAR+right_EAR)/2
            if avg < blink_thresh:
                count_frame += 1  # incrementing the frame count
            else:
                if count_frame >= succ_frame:
                    blink_count += 1
                    
                count_frame = 0

    cam.release()
    cv2.destroyAllWindows()

    normalised_blink_rate = ((blink_count/minutes) - mean_blink_rate) / std_blink_rate

    engagement = 100 - (((normalised_blink_rate+2)/5)*100)

    return int(engagement)

if __name__ == "__main__":
    video_path = './opemipo_attentiveness.mp4'
    test_image = cv2.imread('./opemipo_test.jpeg')

    #register_details('./opemipo_registration.mp4',"Opemipo","Okunoren")
    #print(identify_face(test_image,"Opemipo","Okunoren"))   
    #print(calculate_face_attentiveness(video_path))