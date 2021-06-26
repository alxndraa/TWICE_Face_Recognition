#Grace Vernanda - 2201776552
#Keyzia Alexandra - 2201731573

import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image

def get_path_list(root_path):
    #List of directories inside the trainDir
    trainDirList = os.listdir(root_path)
    folderName = []

    for trainDir in trainDirList:
        folderName.append(trainDir)
        
    return folderName

def get_class_id(root_path, train_names):
    #Read the train images and get the folder id
    images = []
    trainDirID = []

    for index, trainDir in enumerate(train_names):
        imagePathList = os.listdir(root_path + '/' + trainDir)
        for imagePath in imagePathList:
            img = plt.imread(root_path + '/' + trainDir + '/' + imagePath)
            images.append(img)
            trainDirID.append(index)
    
    return images, trainDirID

def detect_train_faces_and_filter(image_list, image_classes_list):
    #detect faces and remove file if no face is found
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayFaceImages = []
    grayFaceID = []
    
    for image, imageID in zip(image_list, image_classes_list):
        if image.dtype != "uint8":
            image = (image * 255).round().astype(np.uint8)
        
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFace = faceCascade.detectMultiScale(imgGray, scaleFactor=1.2, minNeighbors=5, minSize=(100,100))

        if(len(detectedFace) < 1):
            continue
        for faceArea in detectedFace:
            x, y, w, h = faceArea
            #cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
            face = imgGray[y:y+w, x:x+h]
            grayFaceImages.append(face)
            grayFaceID.append(imageID)

    return grayFaceImages, grayFaceID

def detect_test_faces_and_filter(image_list):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    grayFaceImages = []
    faceAreaList = []

    for index, image in enumerate(image_list):
        if image.dtype != "uint8":
            image = (image * 255).round().astype(np.uint8)
            
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFace = faceCascade.detectMultiScale(imgGray, scaleFactor=1.2, minNeighbors=5)
        
        if(len(detectedFace) < 1):
            continue
        for faceArea in detectedFace:
            x, y, w, h = faceArea
            face = imgGray[y:y+w, x:x+h]
            grayFaceImages.append(face)
            faceAreaList.append(faceArea)
    
    return grayFaceImages, faceAreaList

def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    
    return recognizer

def get_test_images_data(test_root_path):
    images = []

    imagePathList = os.listdir(test_root_path)
    
    for imagePath in imagePathList:
        img = plt.imread(test_root_path + '/' + imagePath)
        images.append(img)
    
    return images
    
def predict(recognizer, test_faces_gray):
    resultList = []
    lostList = []

    for image in test_faces_gray:
        result, loss = recognizer.predict(image)
        resultList.append(result)
        lostList.append(loss)
    
    return resultList, lostList

def draw_prediction_results(predict_results, lostList, test_image_list, test_faces_rects, train_names, size):
    drawnPredict = []

    for index, image in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[index]

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        res = predict_results[index]
        loss = lostList[index]
        text = train_names[res] + " (" + str("%.2f" % round(loss, 2)) + ")"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)

        drawnPredict.append(image)
    
    return drawnPredict

def combine_and_show_result(image_list, size):
    combinedImage = Image.new('RGB', (size * len(image_list), size))

    xOffset = 0
    for image in image_list:
        cvtImg = Image.fromarray(np.uint8(image)).convert('RGB')
        combinedImage.paste(cvtImg, (xOffset,0))
        xOffset += size
    combinedImage.show()

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results, lostList = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, lostList, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)