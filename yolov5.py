import torch
import pandas as pd
from xml.dom import minidom
import os 
from os import listdir
from PIL import Image
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
from csv import DictWriter
from pathlib import Path
yolo_list = ['yolov5s','yolov5n','yolov5l']

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generateXML(filename,outputPath,w,h,d,boxes,yolov5,header):
    
    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)
    for i in range(0,len(boxes.ymax)):
         if(boxes.name[i] == "person"):
             childObject = ET.SubElement(top, 'object')
             childName = ET.SubElement(childObject, 'name')
             childName.text = boxes.name[i]
             childPose = ET.SubElement(childObject, 'pose')
             childPose.text = 'Unspecified'
             childTruncated = ET.SubElement(childObject, 'truncated')
             childTruncated.text = '0'
             childDifficult = ET.SubElement(childObject, 'difficult')
             childDifficult.text = '0'
             childConfidence = ET.SubElement(childObject, 'confidence')
             childConfidence.text = str(boxes.confidence[i])
             childBndBox = ET.SubElement(childObject, 'bndbox')
             childXmin = ET.SubElement(childBndBox, 'xmin')
             childXmin.text = str(boxes.xmin[i])
             childYmin = ET.SubElement(childBndBox, 'ymin')
             childYmin.text = str(boxes.ymin[i])
             childXmax = ET.SubElement(childBndBox, 'xmax')
             childXmax.text = str(boxes.xmax[i])
             childYmax = ET.SubElement(childBndBox, 'ymax')
             childYmax.text = str(boxes.ymax[i])
             data = {'Image name':str(filename),'width':width,'height':height,'xmin':boxes.xmin[i],'xmax':boxes.xmax[i],'ymin':boxes.ymin[i],'ymax':boxes.ymax[i],'confidence':boxes.confidence[i],'label':boxes.name[i]}
             with open("/home/bdz1kor/Documents/Models/yolov5/yolov5_csv/" + yolov5 + ".csv","a",encoding = 'UTF8', newline='') as f:
                 dictwriter_object = DictWriter(f, fieldnames=header)
  
                 #Pass the dictionary as an argument to the Writerow()
                 dictwriter_object.writerow(data)
                 f.close()
            
    return prettify(top)


images_folder = '/home/bdz1kor/Documents/Models/yolov5/karthika95-pedestrian-detection/Test/Test/JPEGImages'

header = ['Image name','width','height','xmin','xmax','ymin','ymax','confidence','label']
for i in range(1,4):
    count=1
    model = torch.hub.load('ultralytics/yolov5', yolo_list[i-1], pretrained=True)
    with open('/home/bdz1kor/Documents/Models/yolov5/yolov5_csv/' + yolo_list[i-1] + ".csv",'w', encoding = 'UTF8',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for images in os.listdir(images_folder):

            if (images.endswith(".jpg")):
        

                # Images
                imgs = [images_folder+"/"+images]  # batch of images
                # Inference
                results = model(imgs)
                # Results
                #results.print()
                #results.save()  # or .show()
                results.xyxy[0]  # img1 predictions (tensor)
                df = results.pandas().xywh[0]
                df1 = results.pandas().xyxy[0]
                #df = pd.DataFrame(results.pandas().xyxy[0])
                
                image = Image.open(images_folder+"/"+images)
                
                width = image.width 
                height = image.height
                file = open("/home/bdz1kor/Documents/Models/ensembleObjectDetection/Ensemble/example/"+str(i)+"/"+ images + ".xml", "w")
                file.write(generateXML(images, "/home/bdz1kor/Documents/Models/ensembleObjectDetection/Ensemble/1/", width, height, 3, df1,yolo_list[i-1],header))
                file.close()
            count = count+1
         










