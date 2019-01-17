#!/usr/bin/env python
import sys
import cv2
import time
from random import *
from PIL import Image
import numpy as np
import imutils # for contur
from matplotlib import pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
 
# constants for class Object
IMAGE_SIZE = 200.0
MATCH_THRESHOLD = 3
CASCADEXML=[] # "haarcascade_obj.xml"  # lbpCascade.xml # haarcascade.xml (NOT WORK) # haarcascade_roundabout.xml
CASCADEXML.append("haarcascade_obj.xml") 	# <- TO WORK 0
CASCADEXML.append("lbpCascade.xml")		# <- TO WORK 1

'''
Detect with color
Grundlage: https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
Erweitert auf drei Fabintervalle fuer rote , blaue und gelbe Verkehrszeichen
Ergebnis/Einschaetzung: 
- Vorhandenes Bildmaterial (jpg-Dateien) liefert gute Ergebnisse
- Bei Einsatz in der natuerlichen Umgebung ist eine Bildvorverarbeitung vorzusehen
'''
class ObjectSign:
	def __init__(self):
		print ("hier ist der Objekt-Detektor")
	
	#--- detect Kontur	--------https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/---------------------------
	def detectContur(self, analysebild="objDetect/street/mitKreisverkehr.png"):	
		# Farbgrenzwerte  b, g, r
		#      Verbote (rot), Hauptstr. (gelb), Gebote (blau)
		upLim =[[70, 70, 255], [50, 255, 255], [ 255, 50, 50]]
		lowLim=[[ 0,  0, 70], [ 0,  70,  70], [  70,  0,  0]]
		## Gedanke: Es waere denkbar, die Farbgrenzwerte so zu variieren (Zyklus), dass alle moeglichen Objekte
		##          erkannt herangezogen werden. Aus diesenr Menge wird dan durch ein weiteres Verfahren (deep learning)
		##  		eine weitere Eingrenzung vorgenmommen
		objImages=[] # Liste mit Objekten (Bildausschnitte)
		x0=[]        # Bereich im Gesamtbild
		x1=[]
		y0=[]
		y1=[]
		
		image=cv2.imread(analysebild)  # im bereinigten Bild wird gesucht und ggf. gefundene Objekte entfernt
		objImage = cv2.imread(analysebild) # Bild mit eingezeichneten Regionen
		
		for i in range(0,3): 
			# find the color sign in the image
			upper = np.array(upLim[i])  #max(b,g,r) 
			lower = np.array(lowLim[i])      # min(b,g,r)
			try:
				mask = cv2.inRange(image, lower, upper)
			except:
				print(" -> Bitte Dateiname und Pfad des Analysebildes pruefen!")
				sys.exit(0)
	
			# find contours in the masked image and keep the largest one 
			## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			
				
			try:
				c = max(cnts, key=cv2.contourArea)
				# approximate the contour
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, 0.05 * peri, True)
				# draw a green bounding box surrounding the red game
				#+# cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
				#+# cv2.imshow("detect contur", image)
				# draw Straight Bounding Rectangle
				x,y,w,h = cv2.boundingRect(c)
				xr=int(w*0.1) # Zusaetzlicher Rand 
				yr=int(h*0.1)
				x0.append(x-xr)
				x1.append(x+w+xr)
				y0.append(y-yr)
				y1.append(y+h+yr)
				cv2.rectangle(image,(0,0),(0,0),(0,255,255),0,0)# Dummy, leider erforderlich
			except:
				print("Kein Objekt")
				break
		
		
		if len(x0)<=0:
			print("Sub-Bilder",len(x0))
			print("kein Element gefunden")

		# Uebernahme der Bereiche
		for i in range (0, len(x0)): #len(objImages
			# kleinste Bereiche werden ausgeschlossen
			w=x1[i]-x0[i]
			h=y1[i]-y0[i]
			hw=float(h)/float(w)
			wh=float(w)/float(h) # schliesst trash aus
			if (w>24 and h>24 and hw>0.5 and wh>0.5):
				objImg=image[y0[i]:y1[i],x0[i]:x1[i]] # Objekt-Ausschnitt entnehmen
				objImages.append(objImg)	# Objekt-Ausschnitt der Liste hinzufuegen
				cv2.rectangle(objImage,(x0[i],y0[i]),(x1[i],y1[i]),(0,255,255),2)
				#cv2.waitKey(1)				
		
		'''##Zeigen als Bild [optional zumTesten]
		for i in range (0, len(objImages)): #len(objImages	
			cv2.imshow("object images " + str(i), objImages[i])
			time.sleep(3)
		'''	
  		cv2.imshow("object image ", objImage)	
		cv2.waitKey(3)
	
		return objImage, objImages
'''
Detect with haarcascade
Sucht geometrische Objekte in einem Bild, zum Beispiel Kreise, Rechtecke oder Dreiecke.
Die Basis bzw. die Grundidee entstammt der Internetseiten: 
- https://rdmilligan.wordpress.com/2015/03/01/road-sign-detection-using-opencv-orb/
- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
Weiterbearbeitet und an die konkrete Problemstellung angepasst durch Jonas H.  ab 7.1.2019 .
  Die XML-Datei muss ggf. noch verbessert werden.
'''
class Object:
	def __init__(self):
		print ("hier ist der Objekt-Detektor")
	''' wandelt jpg-Bild in png-Bild um	'''
	def loadJpgSavePng(self, bild):
		im = Image.open(bild + ".jpg")
		im.save(bild + ".png")
	''' Detektiert Strassenschilder in eimem Ausgangsbild und gibt diese als Liste (Feld) von Einzelbildern an das aufrufende Programm zurueck.
	    Zusaetzlich werden die erkannten Objekte im Ausgangsbild umrandet  -- IN WORK haarcascade_sign.xml'''
	def detect(self, analysebild="objDetect/street/mitKreisverkehr.png", objektbild="objDetect/objekt/kreisRotBig.jpg"):
		# load haar cascade and street image # https://github.com/kggreene/sign-detection --> Clonen und anpassen
		signCascade = cv2.CascadeClassifier("objDetect/"+ CASCADEXML[0]) #CASCADEXML[randint(0, 1)]) 
		streetImage = cv2.imread(analysebild)  # Komplettbild
		streetRecImages=cv2.imread(analysebild) # fuer Ausgabe mit Objectrahmen
		# do roundabout detection on street image gray (https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html)
		modStreetImage =streetImage # nicht besser als gray
		
		modStreetImage =cv2.cvtColor(streetImage, cv2.COLOR_RGB2GRAY)  #mod = gray
	
		## https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html
		#-# roundabouts =cv2.CascadeClassifier.detectMultiScale(modSstreetImage, rejectLevels=5,levelWeights=25, scaleFactor=1.4, minNeighbors=3)
		roundabouts = signCascade.detectMultiScale(modStreetImage,  scaleFactor=1.3, minNeighbors=3)
		#roundabouts = signCascade.detectMultiScale(modSstreetImage, scaleFactor=1.4, minNeighbors=3) #standard
		# initialize ORB and BFMatcher
		orb = cv2.ORB()
		bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
		# find the keypoints and descriptors for roadsign image
		objekt = cv2.imread(objektbild) # Das, was gesucht wird <------------- Detect Objekt
		orb = cv2.ORB_create()  #j# Initiate SIFT detector
		kp_r, des_r = orb.detectAndCompute(objekt,None) 
		objImages=[] #++
		# loop through all detected objects
		for (x,y,w,h) in roundabouts:
			# obtain object from street image
			obj = modStreetImage[y:y+h,x:x+w]
			ratio = IMAGE_SIZE / obj.shape[1]
			obj = cv2.resize(obj,(int(IMAGE_SIZE),int(obj.shape[0]*ratio)))
			# find the keypoints and descriptors for object
			kp_o, des_o = orb.detectAndCompute(obj, None)
			### if len(kp_o) == 0 or des_o == None:
			if len(kp_o) == 0: #or des_o == None:
				continue
			# match descriptors
			matches = bf.match(des_r,des_o)
			if(len(matches) >= MATCH_THRESHOLD):  ## mit Schwellwert und +10 fuer etwas groesseren Rahmen als das Objekt selbst
				regionImage=streetImage[y:y+h+10,x:x+w+10] # Objekt-Ausschnitt entnehmen
				objImages.append(regionImage)		# Objekt-Ausschnitt der Liste hinzufuegen
				cv2.rectangle(streetRecImages,(x,y),(x+w+10,y+h+10),(0,255,0),2) # Makieren im Strassenbild
		# show objects on street image
		cv2.imshow('street image in Detector', streetRecImages)
		return streetImage, objImages