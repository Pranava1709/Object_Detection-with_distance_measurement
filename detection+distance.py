	# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:22:14 2020
@author: AryanG
"""

import numpy as np
import os
import six.moves.urllib as urllib
import urllib.request as allib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import pytesseract
import engineio

import torch
from torch.autograd import Variable as V
import models as models
from torchvision import transforms as trn
from torch.nn import functional as F


import pyttsx3
#from .engine import Engine
engine =pyttsx3.init()

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


model_file = 'C:\tensorflow1\models\research\object_detection\inference_graph' 
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

from utils import label_map_util


from utils import visualization_utils as vis_util
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', 'label_map.pbtxt')

NUM_CLASSES = 77


if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


import cv2
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      
      if cv2.waitKey(20) & 0xFF == ord('b'): 
       
          cv2.imwrite('opencv'+'.jpg', image_np) 
      
    
    
          model_file = 'C:\tensorflow1\models\research\object_detection\inference_graph' 
          if not os.access(model_file, os.W_OK):
              weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
              os.system('wget ' + weight_url)
        
          useGPU = 1
          if useGPU == 1:
              model = torch.load(model_file)
          else:
              model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
        
   
          model.eval()
       
          centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
    
        
     
      
          file_name = 'categories_places365.txt'
          if not os.access(file_name, os.W_OK):
              synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
              os.system('wget ' + synset_url)
          classes = list()
          with open(file_name) as class_file:
              for line in class_file:
                  classes.append(line.strip().split(' ')[0][3:])
          classes = tuple(classes)
    
    
        
          img_name = 'opencv.jpg'
          if not os.access(img_name, os.W_OK):
              img_url = 'http://places.csail.mit.edu/demo/' + img_name
              os.system('wget ' + img_url)
    
          img = Image.open(img_name)
          input_img = V(centre_crop(img).unsqueeze(0), volatile=True)
        

          logit = model.forward(input_img)
          h_x = F.softmax(logit, 1).data.squeeze()
          probs, idx = h_x.sort(0, True)
        
          print('POSSIBLE SCENES ARE: ' + img_name)
          engine.say("Possible Scene may be")
          engine.say(img_name)
          
        
          for i in range(0, 5):
              engine.say(classes[idx[i]])
              print('{}'.format(classes[idx[i]]))
      
       
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      
      
      
      
      # Visualization of the results of a detection.
      if cv2.waitKey(2) & 0xFF == ord('a'):
          vis_util.vislize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      else:    
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
      if cv2.waitKey(2) & 0xFF == ord('r'):
          text=pytesseract.image_to_string(image_np)
          print(text)
          engine.say(text)
          engine.runAndWait()
      
    
            
      for i,b in enumerate(boxes[0]):
          


        if classes[0][i] ==43:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("TENNIS IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print("Warning -TENNIS very close to the frame")
                        engine.say("Warning -TENNIS very close to the frame")
        if classes[0][i] ==62:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("chair IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -chair is very close to the frame")
                        engine.say("Warning -chair is very close to the frame")
        if classes[0][i] ==1:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Person is AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print("Warning -Person very close to the frame")
                        engine.say("Warning -Person very close to the frame")

        


        if classes[0][i] ==2:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Bicycle is at a safer distance")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print("Warning -Person very close to the frame")
                        engine.say("Warning -Bicycle very close to the frame")
  
      
        if classes[0][i] == 3:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("car is at a safer distance")

                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                          cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                          print(apx_distance)
                          engine.say(apx_distance)
                          engine.say("units")
                          print("Warning -Car Approaching")
                          engine.say("Warning -Car Approaching")


        if classes[0][i] == 4:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Motorcycle is at a safer distance")


                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance) 
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Motorcycle Approaching")
                        engine.say("Warning -Motorcycle Approaching")
      
        
        if classes[0][i] == 5: 
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("AIRPLANE IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -AIRPLANE very close to the frame")
                        engine.say("Warning -AIRPLANE very close to the frame")
       
        if classes[0][i] ==6:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("bus IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -bus is very close to the frame")
                        engine.say("Warning -bus is very close to the frame")
 
        if classes[0][i] ==7:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("TRAIN IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Train is very close to the frame")
                        engine.say("Warning -Train is very close to the frame")
 
        if classes[0][i] ==8:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("TRUCK IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -TRUCK is very close to the frame")
                        engine.say("Warning -Truck is very close to the frame")
                    
        if classes[0][i] ==8:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("TRUCK IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -TRUCK is very close to the frame")
                        engine.say("Warning -Truck is very close to the frame")
        if classes[0][i] ==9:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("BOAT IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Boat is very close to the frame")
                        engine.say("Warning -Boat is very close to the frame")

        if classes[0][i] ==10:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("traffic light IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -traffic light is very close to the frame")
                        engine.say("Warning -traffic light is very close to the frame")
                
        if classes[0][i] ==11:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("fire hydrant IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -fire hydrant is very close to the frame")
                        engine.say("Warning -fire hydrant is very close to the frame")

        if classes[0][i] ==12:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("FIRE EXTINGUISHER IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -fire extinguisher is very close to the frame")
                        engine.say("Warning -fire extinguisher is very close to the frame")

        if classes[0][i] ==13:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("STOP SIGN IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -stop sign is very close to the frame")
                        engine.say("Warning -stop sign is very close to the frame")

        if classes[0][i] ==14:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("PARKING METER IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Parking Meter is very close to the frame")
                        engine.say("Warning -Parking Meter is very close to the frame")

        if classes[0][i] ==15:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("BENCH IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -bench is very close to the frame")
                        engine.say("Warning -bench is very close to the frame")

        if classes[0][i] ==16:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("BIRD IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Bird is very close to the frame")
                        engine.say("Warning -Bird is very close to the frame")


        if classes[0][i] ==17:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("CAT IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Cat is very close to the frame")
                        engine.say("Warning -Cat is very close to the frame")


        if classes[0][i] ==18:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("DOG IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Dog is very close to the frame")
                        engine.say("Warning -Dog is very close to the frame")


        if classes[0][i] ==19:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("HORSE IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -horse is very close to the frame")
                        engine.say("Warning -horse is very close to the frame")


        if classes[0][i] ==20:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("SHEEP IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -sheep is very close to the frame")
                        engine.say("Warning -sheep is very close to the frame")


        if classes[0][i] ==21:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Cow IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                  if mid_x > 0.3 and mid_x < 0.7:
                      cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                      print(apx_distance)
                      engine.say(apx_distance)
                      engine.say("units")
                      print("Warning -cow is very close to the frame")
                      engine.say("Warning -cow is very close to the frame")


        if classes[0][i] ==22:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Door IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Door is very close to the frame")
                        engine.say("Warning -Door is very close to the frame")


        if classes[0][i] ==23:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("Switchboard IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -Switchboard is very close to the frame")
                        engine.say("Warning -Switchboard is very close to the frame")

   
        if classes[0][i] ==24:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("glass IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -glass is very close to the frame")
                        engine.say("Warning -glass is very close to the frame")

        if classes[0][i] ==25:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("charger IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -charger is very close to the frame")
                        engine.say("Warning -charger is very close to the frame")
                        engine.runAndWait() 

        if classes[0][i] ==26:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("backpack IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -backpack is very close to the frame")
                        engine.say("Warning -backpack is very close to the frame")

        if classes[0][i] ==27:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("umbrella IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -umbrella is very close to the frame")
                        engine.say("Warning -umbrella is very close to the frame")

        if classes[0][i] ==28:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("handbag IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -handbag is very close to the frame")
                        engine.say("Warning -handbag is very close to the frame")

        if classes[0][i] ==29:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("tie IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -tie is very close to the frame")
                        engine.say("Warning -tie is very close to the frame")

        if classes[0][i] ==30:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("teddy bear IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -teddy bear is very close to the frame")
                        engine.say("Warning -teddy bear is very close to the frame")

        if classes[0][i] ==31:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("hair drier IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -hair drier is very close to the frame")
                        engine.say("Warning -hair drier is very close to the frame")

        if classes[0][i] ==32:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("toothbrush IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -toothbrush is very close to the frame")
                        engine.say("Warning -toothbrush is very close to the frame")
                

        if classes[0][i] ==33:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say(" suitcase IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -suitcase is very close to the frame")
                        engine.say("Warning -suitcase is very close to the frame")
                

        if classes[0][i] ==34:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("frisbee IS AT A SAFER DISTANCE")
                 
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -frisbee is very close to the frame")
                        engine.say("Warning -frisbee is very close to the frame")
                

        if classes[0][i] ==35:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("skis IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -skis is very close to the frame")
                        engine.say("Warning -skis is very close to the frame")
                

        if classes[0][i] ==36:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("snowboard IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -snowboard is very close to the frame")
                        engine.say("Warning -snowboard is very close to the frame")
                

        if classes[0][i] ==37:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("sports ball IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -sports ball is very close to the frame")
                        engine.say("Warning -sports ball is very close to the frame")
                

        if classes[0][i] ==38:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("kite IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -kite is very close to the frame")
                        engine.say("Warning -kite is very close to the frame")
                

        if classes[0][i] ==39:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("baseball bat IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -baseball bat is very close to the frame")
                        engine.say("Warning -baseball bat is very close to the frame")
                        engine.runAndWait()                     
                

        if classes[0][i] ==40:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("baseball glove IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -baseball glove is very close to the frame")
                        engine.say("Warning -baseball glove is very close to the frame")

        if classes[0][i] ==41:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("skateboard IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -skateboard is very close to the frame")
                        engine.say("Warning -skateboard is very close to the frame")


        if classes[0][i] ==42:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("surfboard IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -surfboard is very close to the frame")
                        engine.say("Warning -surfboard is very close to the frame")

          

        if classes[0][i] ==44:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("bottle IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -bottle is very close to the frame")
                        engine.say("Warning -bottle is very close to the frame")

        if classes[0][i] ==45:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("vase IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -vase is very close to the frame")
                        engine.say("Warning -vase is very close to the frame")
        if classes[0][i] ==46:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("wine glass IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -wine glass is very close to the frame")
                        engine.say("Warning -wine glass is very close to the frame")

        if classes[0][i] ==47:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("cup IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -cup is very close to the frame")
                        engine.say("Warning -cup is very close to the frame")

        if classes[0][i] ==48:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("fork IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -fork is very close to the frame")
                        engine.say("Warning -fork is very close to the frame")

        if classes[0][i] ==49:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("knife IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -knife is very close to the frame")
                        engine.say("Warning -knife is very close to the frame")
                        engine.runAndWait()

        if classes[0][i] ==50:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("spoon IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -spoon is very close to the frame")
                        engine.say("Warning -spoon is very close to the frame")

        if classes[0][i] ==51:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("bowl IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -bowl is very close to the frame")
                        engine.say("Warning -bowl is very close to the frame")

        if classes[0][i] ==52:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("table IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -table is very close to the frame")
                        engine.say("Warning -table is very close to the frame")

        if classes[0][i] ==53:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("tree IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -tree is very close to the frame")
                        engine.say("Warning -tree is very close to the frame")

        if classes[0][i] ==54:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("printer IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -printer is very close to the frame")
                        engine.say("Warning -printer is very close to the frame")

        if classes[0][i] ==55:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("dustbin IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -dustin is very close to the frame")
                        engine.say("Warning -dustbin is very close to the frame")


        if classes[0][i] ==56:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("stair IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -stair is very close to the frame")
                        engine.say("Warning -stair is very close to the frame")


        if classes[0][i] ==57:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("pen IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -pen is very close to the frame")
                        engine.say("Warning -pen is very close to the frame")

        if classes[0][i] ==58:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("sink IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -sink is very close to the frame")
                        engine.say("Warning -sink is very close to the frame")

        if classes[0][i] ==59:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("refrigerator IS AT A SAFER DISTANCE")
                
                
            if apx_distance <=0.5:
                if mid_x > 0.3 and mid_x < 0.7:
                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    print(apx_distance)
                    engine.say(apx_distance)
                    engine.say("units")
                    print("Warning -refrigerator is very close to the frame")
                    engine.say("Warning -refrigerator is very close to the frame")

        if classes[0][i] ==60:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("book IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -book is very close to the frame")
                        engine.say("Warning -book is very close to the frame")

        if classes[0][i] ==61:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("clock IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -clock is very close to the frame")
                        engine.say("Warning -clock is very close to the frame")

        if classes[0][i] ==62:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("chair IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -chair is very close to the frame")
                        engine.say("Warning -chair is very close to the frame")

        if classes[0][i] ==63:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("couch IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -couch is very close to the frame")
                        engine.say("Warning -couch is very close to the frame")

        if classes[0][i] ==64:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("potted plant IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -potted plant is very close to the frame")
                        engine.say("Warning -potted plant is very close to the frame")

        if classes[0][i] ==65:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("bed IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -bed is very close to the frame")
                        engine.say("Warning -bed is very close to the frame")

        if classes[0][i] ==67:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("dining table IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -dining table is very close to the frame")
                        engine.say("Warning -dining table is very close to the frame")

        if classes[0][i] ==68:
            if scores[0][i] >= 0.5:
                 mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                 mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                 apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                 cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                 print(apx_distance)
                 engine.say(apx_distance)
                 engine.say("units")
                 engine.say("scissors IS AT A SAFER DISTANCE")
                
                
                 if apx_distance <=0.5:
                     if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -scissors is very close to the frame")
                        engine.say("Warning -scissors is very close to the frame")


        if classes[0][i] ==69:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("toaster IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -toaster is very close to the frame")
                        engine.say("Warning -toaster is very close to the frame")


        if classes[0][i] ==70:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("toilet IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -toilet is very close to the frame")
                        engine.say("Warning -toilet is very close to the frame")



        if classes[0][i] ==72:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("tv IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -tv is very close to the frame")
                        engine.say("Warning -tv is very close to the frame")



        if classes[0][i] ==73:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("laptop IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -laptop is very close to the frame")
                        engine.say("Warning -laptop is very close to the frame")




        if classes[0][i] ==74:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("mouse IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -mouse is very close to the frame")
                        engine.say("Warning -mouse is very close to the frame")



        if classes[0][i] ==75:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("remote IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -remote is very close to the frame")
                        engine.say("Warning -remote is very close to the frame")



        if classes[0][i] ==76:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("keyboard IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -keyboard is very close to the frame")
                        engine.say("Warning -keyboard is very close to the frame")

        if classes[0][i] ==77:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("cell phone IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -cell phone is very close to the frame")
                        engine.say("Warning -cell phone is very close to the frame")

        if classes[0][i] ==78:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("microwave IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -microwave is very close to the frame")
                        engine.say("Warning -microwave is very close to the frame")

        if classes[0][i] ==79:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(apx_distance)
                engine.say(apx_distance)
                engine.say("units")
                engine.say("oven IS AT A SAFER DISTANCE")
                
                
                if apx_distance <=0.5:
                    if mid_x > 0.3 and mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        print("Warning -oven is very close to the frame")
                        engine.say("Warning -oven is very close to the frame")
              
          
       
        
                                               
                                   
                
                
            
      
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)
      #cv2.imshow('IPWebcam',image_np)
      cv2.imshow('image',cv2.resize(image_np,(1024,768)))
      if cv2.waitKey(2) & 0xFF == ord('t'):
          cv2.destroyAllWindows()
          cap.release()
          break




#open("yolo-coco/coco.names").read().strip().split("\n")
