#!/usr/bin/env python
import sys
import cv2
import time
from random import *
from PIL import Image, ImageEnhance
import numpy as np
import imutils # for contur
from matplotlib import pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
import imutils # image pyramid
## from pyimagesearch.helpers import pyramid
from skimage.transform import pyramid_gaussian
import argparse


# constants for class Object
IMAGE_SIZE = 200.0
MATCH_THRESHOLD = 3
CASCADEXML=[] # "haarcascade_obj.xml"  # lbpCascade.xml # haarcascade.xml (NOT WORK) # haarcascade_roundabout.xml
CASCADEXML.append("haarcascade_obj.xml") 	# <- TO WORK 0
CASCADEXML.append("lbpCascade.xml")		# <- TO WORK 1

VERBOSE=True # True - Testausgaben, False - keine Testausgaben

'''
Detect with color als Vorverarbeitung
Grundlage: https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
Erweitert auf drei Fabintervalle fuer rote , blaue und gelbe Verkehrszeichen.
Das Bild wird mehrfach durchlaufen , bis kein Objekt mehr gefunden wird.
Ergebnis/Einschaetzung: 
- Vorhandenes Bildmaterial (jpg-Dateien) liefert gute Ergebnisse
- Bei Einsatz in der natuerlichen Umgebung ist eine Bildvorverarbeitung vorzusehen
'''
class ColorFilter:
	def __init__(self):
		print ("hier ist der Farb-Objekt-Detektor ")
		self.comment='' #Instanzvariable

	def setPredictionStr(self, predictionStr):
		predictionNumber, probability, self.comment=predictionStr.data.split("|") # zerlege String
		if VERBOSE: print(predictionNumber, probability, self.comment) # test
		
	'''Strassenbild wird mehrfach durchlaufen, da Schilder eines Typs auch mehrfach vorkommen koennen.
	Wenn kein neues Farbobjekt erkannt wird, wird die Methode beendet
	analysebild - Strassenszene
	return: frameObjektImage - Strassenszene mit Umrandeten Objekten
	return: allObjImages     - Liste mit gefundenen Objekten                                 '''
	def inImages(self, analysebildPfad="objDetect/street/mitKreisverkehr.png"): 
		image=cv2.imread(analysebildPfad, 1)  # im bereinigten Bild wird gesucht und ggf. gefundene Objekte entfernt
		#sizeRoadPicture=(1280,720)
		sizeRoadPicture=(800,450)
		image=cv2.resize(image,sizeRoadPicture)  ## Standardgroesse herstellen
		filledObjImage=image.copy()
		frameObjImage=image.copy()
		# Suchzyklen fuer mehrere Objekte gleichen Farbtyps
		allObjImages=[]
		suchZyklus=0
		while True:
			frameObjImage, filledObjImage, objImages=self.detectContur(image,frameObjImage,filledObjImage) #objImages, u0,v0,u1,v1
			print("Such-Zyklus: %d, Einzelbilder: %d, Name: %s" % (suchZyklus, len(objImages), analysebildPfad))
			# Fuer jeden Suchzyklus Subbilder anhaengen
			for i in range(0,len(objImages)):
				allObjImages.append(objImages[i])
			suchZyklus=suchZyklus + 1
			if len(objImages)<=0:
				break
		cv2.imshow("Strassenbild mit Objektrahmen: "+analysebildPfad, frameObjImage)	
		cv2.waitKey(10)
		return allObjImages, frameObjImage
		
	def inFrame(self, image):
		#sizeRoadPicture=(1280,720)
		sizeRoadPicture=(800,450)
		image=cv2.resize(image,sizeRoadPicture)  ## Standardgroesse herstellen

		filledObjImage=image.copy()
		frameObjImage=image.copy()
		
		# Suchzyklen fuer mehrere Objekte gleichen Farbtyps
		allObjImages=[]
		suchZyklus=0
		while True:
			frameObjImage, filledObjImage, objImages =self.detectContur(image,frameObjImage,filledObjImage)
			print("Such-Zyklus: %d, Einzelbilder: %d, Name: %s" % (suchZyklus, len(objImages), "WebCam"))
			# Fuer jeden Suchzyklus Subbilder anhaengen
			for i in range(0,len(objImages)):
				allObjImages.append(objImages[i])
			
			suchZyklus=suchZyklus + 1
			if len(objImages)<=0:
				break
		#+# cv2.imshow("object frame image: "+"WebCam", frameObjImage)	
		#+# cv2.waitKey(10)
		return  allObjImages, frameObjImage
	
	''' Sucht nach Farbobjekten entsprechend der intern gesetzuten Filtern: rot, gelb, blau.
	Pro Aufruf koennen pro Farbfilter kein oder ein Objekt gefunden werden.
	#--- detect Kontur--- https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
	image - Ausgangsbild, das bleibt unveraendert
	frameObjImage - Bild mit umrandeten Objekten
	filledObjImage - Bild, in dem die bereits gefundenen Objekte getilgt (geloescht) sind.
	return: frameObjImage -  Bild mit weiteren umrandeten Objekten
	return: filledObjImage - Bild, in dem die gefundenen Objekte getilgt sind.
	return: objImages      - gefundene Objekte waehrend dieses Durchlaufs    	'''
	def detectContur(self,image, frameObjImage,filledObjImage):	#image
		# Farbgrenzwerte  b, g, r
		#      Verbote (rot), Hauptstr. (gelb), Gebote (blau) 
		upLim =[[50, 50, 255], [50, 255, 255], [ 255, 50, 50]]
		lowLim=[[ 0,  0,  60], [ 0, 80,   80], [  70,  0,  0]]
		## Gedanke: Es waere denkbar, die Farbgrenzwerte so zu variieren (Zyklus), dass alle moeglichen Objekte
		##          erkannt herangezogen werden. Aus diesenr Menge wird dan durch ein weiteres Verfahren (deep learning)
		##  		eine weitere Eingrenzung vorgenmommen

		x0=[]
		y0=[]		# Bereich im Gesamtbild
		x1=[]
		y1=[]
		for i in range(0,3): 
			# find the color sign in the image
			
			upper = np.array(upLim[i])  #max(b,g,r) 
			lower = np.array(lowLim[i])      # min(b,g,r)
			try:
				mask = cv2.inRange(filledObjImage, lower, upper)
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
				# draw Straight Bounding Rectangle
				x,y,w,h = cv2.boundingRect(c)
				kRand=0.15
				if (i==1): kRand=0.5  # gelbes Schild auf weissem Grund -> Rahmen vergroessert
							 
				xr=int(w * kRand) # Zusaetzlicher Rand 
				yr=int(h * kRand)
				x0.append(x-xr)
				x1.append(x+w+xr)
				y0.append(y-yr)
				y1.append(y+h+yr)
				##cv2.rectangle(filledObjImage,(0,0),(0,0),(0,255,255),0,0)# Dummy, leider erforderlich
			except:
				# print("Kein Objekt")
				break


		# Uebernahme der Bereiche, ausschluss unsinniger Bereiche (zu klein, zu schmal)
		objImages=[] # Liste mit Objekten (Bildausschnitte)
		
		for i in range (0, len(x0)): #len(objImages
			# kleinste Bereiche werden ausgeschlossen
			w=x1[i]-x0[i]
			h=y1[i]-y0[i]
			hw=float(h)/float(w)
			wh=float(w)/float(h) # schliesst trash aus
			if (w>31 and h>31 and hw>0.5 and wh>0.5):
				cv2.waitKey(100) # notwendig zur Synchronisation der Datenstroeme
				objImg=image[y0[i]:y1[i],x0[i]:x1[i]] # Objekt-Ausschnitt entnehmen
				 #t# print("%d,%d,%d, %d"% (y0[i],y1[i],x0[i],x1[i]))
				objImages.append(objImg)	# Objekt-Ausschnitt der Liste hinzufuegen
				cv2.rectangle(frameObjImage,(x0[i],y0[i]),(x1[i],y1[i]),(upLim[i]),2)
				cv2.rectangle(frameObjImage,(x0[i]-2,y0[i]-2),(x1[i]+2,y1[i]+2),(lowLim[i]),2)
				cv2.rectangle(filledObjImage,(x0[i],y0[i]),(x1[i],y1[i]),(255,255,255),cv2.FILLED)
				if VERBOSE: cv2.imshow("subimage: "+str(i), objImages[i])
	
		return frameObjImage, filledObjImage, objImages     
		

			
		
