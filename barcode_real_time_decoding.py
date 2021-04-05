# import the necessary packages
from __future__ import print_function
import numpy as np
from imutils.video import VideoStream
import argparse
import time
import cv2
import pyzbar.pyzbar as pyzbar
import os
import glob
import pandas as pd
height=0;
width=0;
layers=0;

###############################################################################

def decode(im, i) : 
# Find barcodes 
  decodedObjects = pyzbar.decode(im)
  df=pd.read_csv("trial.csv") 
  val=df['Bar_code']
  indent = 1;

  font = cv2.FONT_HERSHEY_SIMPLEX
  img = np.zeros((1080,1920,3), np.uint8)

  for obj in decodedObjects:
    code= str(obj.data)
    code=code[2:16]
    print('Barcode Number: ', code,'\n')
    result = ""
    for j in range(len(val)):
      if str(val[j])==code:
        result = result + "Product Name : " + df['Product_Name'][j]
        result = result + "   Manufacturer Name : " + df['Manufacturer details'][j]
        cv2.putText(img,result,(10,indent*50), font, 1,(255,255,255),2)
        indent = indent+1

        result = ""
        result = result + "Price : " + str(df['Price'][j])
        result = result + "   Net Weight : " + str(df['Net_wt'][j])
        cv2.putText(img,result,(10,indent*50), font, 1,(255,255,255),2)
        indent = indent+3

# Drawing bounding box on the image
  for decodedObject in decodedObjects: 
   
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
    
    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

  # Stack the image and result vertically and save
  height, width, layers = im.shape

  img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)  
  name = "./imgs/frame_" + str(i) + ".jpg"
  collage = np.vstack([im, img])
  collage = cv2.resize(collage, (480, 540), interpolation = cv2.INTER_AREA)
  cv2.imwrite(name, collage);

###############################################################################

image_folder = 'imgs'
files = glob.glob('./'+image_folder+'/*')
for f in files:
  os.remove(f)
i=0;
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())
# if the video path was not supplied, grab the reference to the
# camera
if not args.get("video", False):
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, load the video
else:
	vs = cv2.VideoCapture(args["video"])


# keep looping over the frames
while True:
  # grab the current frame and then handle if the frame is returned
  # from either the 'VideoCapture' or 'VideoStream' object,
  # respectively
  frame = vs.read()
  frame = frame[1] if args.get("video", False) else frame
  
  # check to see if we have reached the end of the
  # video
  if frame is None:
  	break
  frame = cv2.resize(frame, (1280, 720))
  decode(frame, i)
  i=i+1

video_name = 'decoded_video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

sample = cv2.imread(os.path.join(image_folder, 'frame_1.jpg'))
height, width, channels = sample.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
