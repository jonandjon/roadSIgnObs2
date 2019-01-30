#!/usr/bin/env python
import cv2
import roslib
import rospy
import tensorflow as tf
import keras
import numpy as np
from keras import backend as k
from keras import callbacks		
from keras.datasets import mnist 
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, String, Int16MultiArray
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dense, Dropout, Flatten 
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from PIL import Image
import time
import os
import tensorflow as tf
import Cnn       # mein Modul mit Klassen
## from thread import start_new_thread 
## from threading import Thread
OBJ_ROWS, OBJ_COLS= 32, 32 # input image dimensions
''' Instanz der Klasse: Verkehrsschilder Objekt=Modul.Klasse()   '''
cnn=Cnn.Gtsrb(OBJ_ROWS, OBJ_COLS) 

''' Kommunikation und Aufruf der Vorhersageklasse (CNN): '''
class Prediction:
	''' Konstruktor	 '''	
	def __init__(self):
	
		self.cv_bridge = CvBridge()

		print("Hier ist der Prediction-Konstruktor")
		
						 
		## publish back the prediction images '/camera/input/specific/comment
		self.publisherPrediction = rospy.Publisher("/camera/output/specific/prediction",
								 String,
								 queue_size=1)								
		
		## publish back the prediction images
		self.publisherPredictionImage = rospy.Publisher("/camera/output/specific/compressed_img_msgs",
								 CompressedImage,
								 queue_size=1)
	

		# Subscriber der fuer Objektbilder 32 x32 Pixel -------------------------
		self. subscribCam = rospy.Subscriber('/camera/output/webcam/compressed_img_msgs',
							CompressedImage,
							self.callbackObjImage,
							queue_size = 1)
		
							
	''' Verarbeitet ein Objektbild 32x32 Pixel '''											
	def callbackObjImage(self, Cam):
		# print ('CALLBECK: object images of type: "%s"' % Cam.format)
		np_arr = np.fromstring(Cam.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR ) #cv2.CV_LOAD_IMAGE_COLOR
		np_image=img_to_array(image_np, data_format='channels_last')
		(h,w,c) = np_image.shape
		if h>OBJ_ROWS: return
		
		print("Objektbild im Format (shape): ", h, w, c)
		## Bildbewertung
		prediction, probability = cnn.predictImage(np_image)
		
		predictionComment='SICHER' ## Wertung		
		if probability <0.97: predictionComment = "UNSICHER"
		if probability <0.90: predictionComment = "SEHR UNSICHER"
		if probability <0.85: predictionComment = "TRASH"
		predictionComment="%13s (p: %-5.3f):" % (predictionComment, probability,)
		#+
		prediction= str(prediction) + "|"+ predictionComment # Eine Uebertragung -> SYNCHRON
		# Sende Prediction-Label und das dazugehoerige Bild zurueck #
		self.publisherPrediction.publish(prediction)
		#t# cv2.imshow('object images in Prediction', image_np) ## test
		self.publisherPredictionImage.publish(Cam) # wird nur weitergereicht
		cv2.waitKey(10)
	
	
	''' Methoden der Klasse
	Hilfsmethoden zum Speichern'''
	
	def saveArry(self, nparray, name="nparray"):
		image=array_to_img(nparray, data_format = "channels_last")
		image.save("./resultImg/"+name+".jpg")
		
	def saveImg(self, image, name="nparray"):
		image.save("./resultImg/"+name+".jpg")
	
	# Speichert ein Bild der Trainings-Menge mit Index'''
	def saveBild(self, imageIndex):  ## for test
		(self.imagesTrain, self.labelsTrain), (self.imagesTest, self.labelsTest) = mnist.load_data()
		npArrayBild=np.expand_dims(self.imagesTest[imageIndex], axis=0) 
		label=self.labelsTest[imageIndex]
		## wandle in ein Bild um
		bild=Image.fromarray(npArrayBild[0])
		## speichert das Bild mit dem Dateinamen
		zeit=time.strftime("%H:%M:%S")
		filenameTrain="B"+str(imageIndex) +"L"+str(label)+"_"+zeit+".jpg"
		bild.save(filenameTrain)
		bild.close()
	''' -------------------------------------------------------------------------------------------- '''
def main():
		print("try in main")
		# register node
		rospy.init_node('prediction', anonymous=False)
		# init Prediction
		pred = Prediction()
		# TESTREGION: TESTEN MIT BEKANNTEM BILD 
		##  Bild aus der Trainingnge wird ausgewaehlt 
		imageIndex=7 # wie in Camera Pseudo (es ist uebrigend die Ziffer 4 die zu erkennen ist)
		### Trainingsmodell, DEEP LEARNING TRAINING 
		### TEST:Predict an Images and load Trainimages
		inputLabel, predictionLabel= cnn.predictTestImage(imageIndex)
		print("Bereit zum Empfang der Bilder zur Prediction ... ")
		try:
			rospy.spin()
		except KeyboardInterrupt:
			print "Shutting down ROS SubsciberCam"
		cv2.destroyAllWindows()	

if __name__ == '__main__':
	main()
