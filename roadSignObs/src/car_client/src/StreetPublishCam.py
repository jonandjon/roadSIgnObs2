#!/usr/bin/env python
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
#-# from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from PIL import Image
# from threading import Thread
import threading
import time
import sys
import os
import imageio
import csv ###
import random
import time 

img_rows, img_cols = 32, 32  # input image dimensions
PAUSE = 10000 # Millisekunden
PUBLISH_RATE = 3 # fuer WebCam in Hz
USE_WEBCAM=False # True: WebCam, False: Strassenszenen aus dem Verzeichnis street

#Klasse zum Normen der hineingefuegten Bilder in dem Street -Ordner
class NormImages:
	def __init__(self):
		print ("NormImages only")
	'''fuer Strassenbilder'''
	def inImages(self, analysebildPfad="objDetect/street/mitKreisverkehr.png"): 
		image=cv2.imread(analysebildPfad, 1)  # im bereinigten Bild wird gesucht und ggf. gefundene Objekte entfernt
		#sizeRoadPicture=(1280,720)
		sizeRoadPicture=(800,450)
		image=cv2.resize(image,sizeRoadPicture)  ## Standardgroesse herstellen
		images=[]
		images.append(image) #moeglicherweise kommt noch etwas dazu
		#t# cv2.imshow("only image: "+analysebildPfad, image)
		return images
		
	'''In einer Webcam/Video'''
	def inFrame(self, image): 
		#sizeRoadPicture=(1280,720)
		sizeRoadPicture=(800,450)
		image=cv2.resize(image,sizeRoadPicture)  ## Standardgroesse herstellen
		images=[]
		images.append(image) #moeglicherweise kommt noch etwas dazu
		#t# cv2.imshow("only image from WebCam: ", image)
		return images		

# Instanz der Klasse ...
objectSign=NormImages()

class PublishWebCam:
	#Konstruktor 
	def __init__(self):
		self.cv_bridge = CvBridge()
		# publish webcam
		self.publisher_webcam_comprs = rospy.Publisher("/camera/output/webcam/compressed_img_msgs",
                                                       CompressedImage,
                                                       queue_size=1)
													   
		# publish Frame nur eine publisher von noetig 
		self.publisher_fullcam_comprs = rospy.Publisher("/fullcamera/output/webcam/compressed_img_msgs",
                                                       CompressedImage,
                                                       queue_size=1)												

		if USE_WEBCAM==True:
			self.input_stream = cv2.VideoCapture(0)
			if not self.input_stream.isOpened():
				raise Exception('Camera stream did not open\n')
        rospy.loginfo("Publishing data...")
#--------------------------------------------------------------------------------------
	''' liest ein zufaellige Strassen-Bilddateien '''
	def readRoadPictures(self, rootpath="./objDetect/street/"):
		namesPictures = [] # images
		gtFile = open(rootpath + '/roadPictures.csv') # csv-Datei enthaelt Namen der zur Auswahl stehenden Bilddateien
		gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
		gtReader.next() # skip header
		# loop over all images in current annotations file
		for row in gtReader:
			dateiname=rootpath + row[0]
			namesPictures.append(dateiname)
		gtFile.close()
		return namesPictures 
		
	''' veroeffentlicht Daten '''
	def cam_data(self, verbose=0):
		rate = rospy.Rate(PUBLISH_RATE)  #Takrate wie oft ausgefuehrt wird?
		while not rospy.is_shutdown():
			# reactivate for webcam image. Pay attention to required subscriber buffer size.
			# See README.md for further information
			if USE_WEBCAM==True:
				print("WEBCAM is true!")
				# Methode zum veroeffentlichen des Vollbildes 
				camFrame=self.getCamFrame(verbose)
				rate.sleep() 
				allObjImages=objectSign.inFrame(camFrame)
				#fuer TEST# cv2.imwrite("objDetect/street/camFrame.png", camFrame)  #+++
				#         # cv2.imshow('camFrame in PublishCam', camFrame)
			else:  	# Strassenszenen aus Verzeichnis objDetect/street
				namesPictures=self.readRoadPictures() #rootpath="./TestImages"
				zufallsindex=random.randint(0, len(namesPictures)-1) #+++
				# zufallsindex=3 # immer das selbe Bild
				Images=objectSign.inImages(namesPictures[zufallsindex])  #
			for img in Images: ## koennten auch mehrere Bilder in einer Liste sein ###
				#+# self.saveAsPPM(npImage=img, pfad='ABLAGE/') # zum Testen
				try:
					print("image publish", (int(time.time()))) # Kontrollausgabe
					compressed_imgmsg = self.cv_bridge.cv2_to_compressed_imgmsg(img)
					self.publisher_webcam_comprs.publish(compressed_imgmsg)
					cv2.imshow('camFrame in PublishCamOnly', img)
				except:
					print("kein gueltiges Bild") 			
				cv2.waitKey(PAUSE) 
				#time.sleep(PAUSE/1000)
				cv2.destroyAllWindows() #*#
	
	''' Sendet Vollbilder der Webcam fortlaufend 
	    OPTIONAL  '''
	def getCamFrame(self, verbose=0):
		if self.input_stream.isOpened() and USE_WEBCAM:
			success, frame = self.input_stream.read()
			msg_frame = self.cv_bridge.cv2_to_compressed_imgmsg(frame)
			# -> Uebertragung der vollstaendigen-Camera-Bilder
			#OPTIONAL# self.publisher_fullcam_comprs.publish(msg_frame.header, msg_frame.format, msg_frame.data)
			if verbose:
				rospy.loginfo(msg_frame.header.seq)
				rospy.loginfo(msg_frame.format)
		return frame
	
	'''Debugmethode Speichert ein nymphi-array als ppm-Bild '''	
	def saveAsPPM(self, npImage, pfad='ABLAGE/img.ppm' ):
		try:
			b, g, r = cv2.split(npImage)
			npImage=cv2.merge((r,g,b))
			now=str(time.time())
			datei=pfad + 'img'+now+'.ppm'
			#t# cv2.imwrite('ABLAGE/img'+ now +'.ppm', img) #CV_IMWRITE_PXM_BINARY
			# imageio.imwrite(datei, npImage, format='PPM-FI', flags=1) #'PPM-FI' (ascii)
			imageio.imwrite(datei, npImage, format='PPM') #'PPM-FI' (ascii)
		except:
			print("kein Objektbild") 
		
	
def main():
	verbose = 0  # use 1 for debug
	# register node
	rospy.init_node('PublishWebCam', anonymous=False)
	# Instanz der Klasse (Publisher)
	cam = PublishWebCam()
	# start publishing data
	cam.cam_data(verbose)  #Veroeffentlich Daten an ein Topic
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
        	pass
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
