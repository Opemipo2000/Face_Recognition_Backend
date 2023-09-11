from django.conf import settings
from django.shortcuts import render

# Create your views here.
import numpy as np
import pandas as pd
import cv2
import os
import pickle
import uuid
import random
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import FaceModel
from .serializers import FaceModelSerializer
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

    
class RegisterFace(APIView):
    def post(self, request, *args, **kwargs):
        video = request.FILES['video']
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        email = request.data.get('email')
        user_id = uuid.uuid4()
        
        vidcap = cv2.VideoCapture(video.temporary_file_path())
        headshot_dir = f"Headshots/{first_name}_{last_name}/"
        headshot_img_dir = f"Headshots/{first_name}_{last_name}/images/{user_id}.jpg"
        os.makedirs(os.path.join(settings.MEDIA_ROOT, headshot_dir), exist_ok=True)
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, headshot_img_dir), image)     # save frame as JPG file
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
            
        # images_dir = "./Headshots"
        current_id = 0
        label_ids = {}
        for root, _, files in os.walk(headshot_img_dir):
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

        train_generator = train_datagen.flow_from_directory(headshot_img_dir, target_size=(224,224), color_mode='rgb', batch_size=32, class_mode='categorical', shuffle=True)

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
        h5_folder_dir = f"h5_folder/{first_name}_{last_name}_{user_id}_transfer_learning_trained_face_cnn_model.h5" 
        model.save(os.path.join(settings.MEDIA_ROOT, h5_folder_dir))

        # deletes the existing model
        del model
        
        # saving training labels
        class_dictionary = train_generator.class_indices
        class_dictionary = {
            value:key for key, value in class_dictionary.items()
        }
        
        face_instance = FaceModel(
        user_id=user_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        h5_model=h5_folder_dir,
        face_labels=class_dictionary,
        headshot_directory=headshot_dir
        )
        face_instance.save()
        
        # Serialize and return the instance.
        serializer = FaceModelSerializer(face_instance)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
        

class IdentifyFace(APIView):
    def image_preprocessor(self, new_image):
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        new_image = cv2.normalize(new_image, None, alpha=0, beta=1.5*255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        faces_coordinates = face_detector.detectMultiScale(gray_img)

        for (x, y, w, h) in faces_coordinates:
            face = new_image[y:y+h, x:x+w]
            face = cv2.resize(face, [224, 224])
            face = image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = utils.preprocess_input(face, version=1)
            return face

    def post(self, request, format=None):
        first_name = request.data.get('first_name')
        last_name = request.data.get('last_name')
        email = request.data.get('email')
        image_file = request.data.get('image_file')  # Assuming the image is sent as a file
        
        user_id = FaceModel.objects.filter(email=email)

        # Convert uploaded image file to a numpy array
        np_arr = np.frombuffer(image_file.read(), np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        processed_image = self.image_preprocessor(image_np)

        # Load the model and predict
        model_path = os.path.join(settings.MEDIA_ROOT, f'./h5_folder/{first_name}_{last_name}_{user_id}_transfer_learning_trained_face_cnn_model.h5')
        if not os.path.exists(model_path):
            return Response({"message": "Model does not exist for this user."}, status=status.HTTP_400_BAD_REQUEST)
        
        model = load_model(model_path)

        try:
            label_obj = FaceModel.objects.get(first_name=first_name, last_name=last_name)
            class_dictionary = label_obj.label_data
        except FaceModel.DoesNotExist:
            return Response({"message": "Face label data does not exist for this user."}, status=status.HTTP_400_BAD_REQUEST)

        class_list = [value for _, value in class_dictionary.items()]

        # Perform prediction
        test_preds = model.predict(processed_image)
        identity = class_list[test_preds[0].argmax()]

        return Response({"identity": identity}, status=status.HTTP_200_OK)
