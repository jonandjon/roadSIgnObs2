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

''' Instanz der Klasse: Verkehrsschilder Objekt=Modul.Klasse()   '''
cnn=Cnn.Gtsrb() 

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
	

		# Subscriber der Kamerabilderoder der Validierungsbilder -------------------------
		self. subscribCam = rospy.Subscriber('/camera/output/webcam/compressed_img_msgs',
							CompressedImage,
							self.callbackCam,
							queue_size = 1)
												
	def callbackCam(self, Cam):
		#-# if VERBOSE:
		print ('CALLBECK: received image of type: "%s"' % Cam.format)
		np_arr = np.fromstring(Cam.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR ) #cv2.CV_LOAD_IMAGE_COLOR
		## Bildbewertung
		np_image=img_to_array(image_np, data_format='channels_last')
		predictionLabel, predictionComment= cnn.predictImage(np_image)
		prediction= str(predictionLabel) + "|"+ predictionComment # Eine Uebertragung -> SYNCHRON
		# Sende Prediction-Label und das dazugehoerige Bild zurueck #
		self.publisherPrediction.publish(prediction)
		#test#  cv2.imshow('Prediction img', image_np) ## test
		self.publisherPredictionImage.publish(Cam)
		cv2.waitKey(10)

	''' Methoden der Klasse
	## a) Hilfsmethoden
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
		imageIndex=6 # wie in Camera Pseudo (es ist uebrigend die Ziffer 4 die zu erkennen ist)
		### Trainingsmodell, DEEP LEARNING TRAINING 
		### Predict an Images and load Trainimages
		inputLabel, predictionLabel= cnn.predictTestImage(imageIndex)
		print("Bereit zum Empfang der Bilder zur Prediction ... ")
		try:
			rospy.spin()
		except KeyboardInterrupt:
			print "Shutting down ROS SubsciberCam"
		cv2.destroyAllWindows()	

if __name__ == '__main__':
	main()
