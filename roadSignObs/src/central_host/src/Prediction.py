#!/usr/bin/env python
import cv2
import roslib
import rospy
import tensorflow as tf
import keras
import numpy as np
from keras import backend as k
from keras import callbacks		## ++j
from keras.datasets import mnist  ## for Test
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, String
from keras.layers import Conv2D, MaxPooling2D ##++
from keras.layers import Dense, Dropout, Flatten ## ++
from keras.models import Sequential ## ++
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
## j.h add ##
from PIL import Image
import time
import os
## os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import Cnn       # mein Modul mit Klassen
## from thread import start_new_thread 
## from threading import Thread

''' Instanz der Klasse: Verkehrsschilder Objekt=Modul.Klasse()   '''
cnn=Cnn.Gtsrb() 

''' Vorhersageklasse: '''
class Prediction:
	''' Konstruktor	 '''	
	def __init__(self):
	
		self.cv_bridge = CvBridge()

		print("Hier ist der Prediction-Konstruktor")
		
		
		## Publisher sendet Label des erkannten Bildes
		self.publisherPredictionNumber = rospy.Publisher("/camera/input/specific/number",
															 Int32,			#prediction-Label
															 queue_size=1)
															 
		## publish back the prediction images
		self.publisherPredictionImage = rospy.Publisher("/camera/output/specific/compressed_img_msgs",
														 CompressedImage,
														 queue_size=1)


									
		# Subscriber der Kamerabilderoder der Validierungsbilder
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
		npImage=img_to_array(image_np, data_format='channels_last')
		predictionLabel= cnn.predictImage(npImage)
		# Zeige Bildfolge -> cv_img
		#++# cv2.imshow('cv_img', image_np)  ## nicht auf Windows --> ubuntu notwendig
		# Sende Prediction-Label und das dazugehoerige Bild zurueck #
		self.publisherPredictionNumber.publish(predictionLabel)
		self.publisherPredictionImage.publish(Cam)
		cv2.waitKey(1)
		'''
		KeyboardInterrupt:
			print "Shutting down ROS SubsciberCam"
		'''

	''' Methoden der Klasse
	## a) Hilfsmethoden
	# Speichert ein Bild der Trainings-Menge mit Index	'''
	def saveBild(self, imageIndex):  ## for test
		(self.imagesTrain, self.labelsTrain), (self.imagesTest, self.labelsTest) = mnist.load_data()
		npArrayBild=np.expand_dims(self.imagesTest[imageIndex], axis=0) 
		label=self.labelsTest[imageIndex]
		## wandle in ein Bild um
		bild=Image.fromarray(npArrayBild[0])
		# speichert das Bild mit dem Dateinamen
		zeit=time.strftime("%H:%M:%S")
		filenameTrain="B"+str(imageIndex) +"L"+str(label)+"_"+zeit+".jpg"
		bild.save(filenameTrain)
		bild.close()
    	
	''' * Kommunikationsmethoden
        * vom Subscriber mit dem topic: /camera/output/specific/check
          um das Validierungsergebnis zu empfangen '''
	#-# def callbackVerifyPrediction(self, verify):
 		#-# print("callback verify : %s" % (verify,))

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
		'''
		print("--- print in prediction.py -----------------")
		print(" Index des zu identifizierenden Bildes: %s" % (imageIndex,)) 
		print(" Label des zu identifizierenden Bildes: %s" % (inputLabel,))
		print(" Label des prediction Bild:           : %s" % (predictionLabel,))
		print("=============================================")
		# Publish your predicted number
	    ## zum Test und zur Anschauung
		### pred.saveBild(imageIndex) ''' # Ende TESTREGION
		
		print("Bereit zum Empfang der Bilder zur Prediction ... ")
		try:
			rospy.spin()
		except KeyboardInterrupt:
			print "Shutting down ROS SubsciberCam"
			#-# rospy.ROSInterruptException:
			#-# pass
		cv2.destroyAllWindows()	


if __name__ == '__main__':
	main()
