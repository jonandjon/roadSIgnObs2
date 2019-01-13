#!/usr/bin/env python
'''
Sucht geometrische Objekte in einem Bild, zum Beispiel Kreise, Rechtecke oder Dreiecke.
Die Basis bzw. die Grundidee entstammt der Internetseiten: 
- https://rdmilligan.wordpress.com/2015/03/01/road-sign-detection-using-opencv-orb/
- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
Weiterbearbeitet und an die konkrete Problemstellung angepasst durch Jonas H.  ab 7.1.2019 .
  Die XML-Datei muss ggf. noch verbessert werden.
 '''

import cv2
import time
from random import *
from PIL import Image
 
# constants
IMAGE_SIZE = 200.0
MATCH_THRESHOLD = 3
CASCADEXML=[] # "haarcascade_obj.xml"  # lbpCascade.xml # haarcascade.xml (NOT WORK) # haarcascade_roundabout.xml
CASCADEXML.append("haarcascade_obj.xml") 	# <- TO WORK 0
CASCADEXML.append("lbpCascade.xml")		# <- TO WORK 1

class Object:
	def __init__(self):
		print ("hier ist der Objekt-Detektor")
	''' wandelt jpg-Bild in png-Bild um	'''
	def loadJpgSavePng(self, bild):
		im = Image.open(bild + ".jpg")
		im.save(bild + ".png")
	''' Detektiert Strassenschilder in eimem Ausgangsbild und gibt diese als Liste (Feld) von Einzelbildern an das aufrufende Programm zurueck.
	    Zusaetzlich werden die erkannten Objekte im Ausgangsbild umrandet  -- IN WORK haarcascade_sign.xml'''
	def detect(self, analysebild="objDetect/street/mitKreisverkehr.png", objektbild="objDetect/objekt/kreisKlein.jpg"):
		# load haar cascade and street image # https://github.com/kggreene/sign-detection --> Clonen und anpassen
		signCascade = cv2.CascadeClassifier("objDetect/"+ CASCADEXML[0]) #CASCADEXML[randint(0, 1)]) 
		streetImage = cv2.imread(analysebild) # Komplettbild 
		streetRecImages=cv2.imread(analysebild) # fuer Ausgabe mit Objectrahmen
		# do roundabout detection on street image gray
		grayStreetImage =cv2.cvtColor(streetImage, cv2.COLOR_RGB2GRAY) 
		#T# cv2.imshow('grayStreetImage ... ', grayStreetImage)
		## https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html
		#-# roundabouts =cv2.CascadeClassifier.detectMultiScale(grayStreetImage, rejectLevels=5,levelWeights=25, scaleFactor=1.4, minNeighbors=3)
		roundabouts = signCascade.detectMultiScale(grayStreetImage,  scaleFactor=1.3, minNeighbors=3)
		#roundabouts = signCascade.detectMultiScale(grayStreetImage, scaleFactor=1.4, minNeighbors=3) #standard
		# initialize ORB and BFMatcher
		orb = cv2.ORB()
		bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
		# find the keypoints and descriptors for roadsign image
		objekt = cv2.imread(objektbild) # Das, was gesucht wird <------------- Detect Objekt
		## cv2.imshow('roadsign', objekt) #j# 
		## cv2.waitKey(1000)
		orb = cv2.ORB_create()  #j# Initiate SIFT detector
		kp_r, des_r = orb.detectAndCompute(objekt,None) 
		objImages=[] #++
		# loop through all detected objects
		for (x,y,w,h) in roundabouts:
			# obtain object from street image
			obj = grayStreetImage[y:y+h,x:x+w]
			ratio = IMAGE_SIZE / obj.shape[1]
			obj = cv2.resize(obj,(int(IMAGE_SIZE),int(obj.shape[0]*ratio)))
			# find the keypoints and descriptors for object
			kp_o, des_o = orb.detectAndCompute(obj, None)
			### if len(kp_o) == 0 or des_o == None:
			if len(kp_o) == 0: #or des_o == None:
				continue
			# match descriptors
			matches = bf.match(des_r,des_o)
			# draw object on street image, if threshold met
			#+# cv2.rectangle(streetImage,(x,y),(x+w,y+h),(255,0,0),1) # alle Objekte
			if(len(matches) >= MATCH_THRESHOLD):  ## mit Schwellwert und +10 fuer etwas groesseren Rahmen als das Objekt selbst
				regionImage=streetImage[y:y+h+10,x:x+w+10] # Objekt-Ausschnitt entnehmen
				objImages.append(regionImage)		# Objekt-Ausschnitt der Liste hinzufuegen
				cv2.rectangle(streetRecImages,(x,y),(x+w+10,y+h+10),(0,255,0),2) # Makieren im Strassenbild
		# show objects on street image
		cv2.imshow('street image in Detector', streetRecImages)
		## cv2.waitKey(10000)
		## cv2.destroyAllWindows()
		return streetImage, objImages