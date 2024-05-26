import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

from pymongo import MongoClient

from dotenv import load_dotenv
load_dotenv()

dbclient = MongoClient()
db = dbclient.dipl

main_dir =os.environ["MAIN_FOLDER"]
folder_dir = os.path.join(main_dir, "NAPS_L")

font = cv2.FONT_HERSHEY_COMPLEX

Threshold = 0.5
image_size = 320


def final_prediction(prediction_box , bounding_box , confidence , class_labels,width_ratio,height_ratio):
    obj_list = []
    if len(prediction_box) > 0:
        for j in prediction_box.flatten():
            x, y , w , h = bounding_box[j]
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            bb_dict = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }

            label = str(classes_names[class_labels[j]])
            conf_ = str(round(confidence[j],2))
            obj_dict = {
                "name": label,
                "bounding_box": bb_dict,
                "confidence": conf_
            }
            obj_list.append(obj_dict)
        
        # cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,0,255) , 2)
        # cv2.putText(image , label+' '+conf_ , (x , y-2) , font , .2 , (0,255,0),1)

    return obj_list

def bounding_box_prediction(output_data):
    bounding_box = []
    class_labels = []
    confidence_score = []
    for i in output_data:
        for j in i:
            high_label = j[5:]
            classes_ids = np.argmax(high_label)
            confidence = high_label[classes_ids]
            
            if confidence > Threshold:
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
                x , y = int(j[0] * image_size - w/2) , int(j[1] * image_size - h/2)
                bounding_box.append([x,y,w,h])
                class_labels.append(classes_ids)
                confidence_score.append(confidence)

    
    prediction_boxes = cv2.dnn.NMSBoxes(bounding_box , confidence_score , Threshold , .6)    
    return prediction_boxes , bounding_box ,confidence_score,class_labels




Neural_Network = cv2.dnn.readNetFromDarknet(os.path.join(main_dir, "darknet-yolo/darknet/cfg/yolov3-openimages.cfg"), os.path.join(main_dir, "darknet-yolo/yolov3-openimages.weights"))


classes_names = []
k = open(os.path.join(main_dir, "darknet-yolo/darknet/data/openimages.names"),'r')
for i in k.readlines():
    classes_names.append(i.strip())


for jpg in os.listdir(folder_dir):
    image = cv2.imread(os.path.join(folder_dir, jpg))
    original_with , original_height = image.shape[1] , image.shape[0]

    mongo_dict = {
        "title": jpg
    }

    blob = cv2.dnn.blobFromImage(image , 1/255 , (320,320) , True , crop = False)
    Neural_Network.setInput(blob)
    cfg_data = Neural_Network.getLayerNames()
    layer_names = Neural_Network.getUnconnectedOutLayers()
    outputs = [cfg_data[i-1] for i in layer_names]
    output_data = Neural_Network.forward(outputs)

    prediction_box , bounding_box , confidence , class_labels = bounding_box_prediction(output_data)

    mongo_dict["objects"] = final_prediction(prediction_box , bounding_box , confidence , class_labels , original_with / 320 , original_height / 320 )


    result = db.openimages.insert_one(mongo_dict)
    print('Inserted post id %s with name %s ' % (result.inserted_id, jpg))
    # cv2.imshow('image',image)
    # cv2.waitKey(0)