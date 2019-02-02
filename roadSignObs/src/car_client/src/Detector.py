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
import time

VERBOSEprio=0 # 0 - keine Testausgaben ... 10 - alle Testausgaben
'''
Datenstruktur der Bildobjekte (Bildausschnitte)
'''
class Objekt:
	def __init__(self):
		self.images=[]  #Bildausschnitte
		self.x0=[]	# Koordinaten in Bezug zum Strassenbild (Gesamtbild)	
		self.y0=[]	# ...
		self.x1=[]
		self.y1=[]

'''
Vorverarbeitung mit Hilfe von Farbfiltern
Grundlage: https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
Erweitert auf drei Fabintervalle fuer rote , blaue und gelbe Verkehrszeichen.
Das Bild wird mehrfach durchlaufen , bis kein Objekt mehr gefunden wird.
Ergebnis/Einschaetzung: 
- Vorhandenes Bildmaterial (jpg-Dateien) liefert gute Ergebnisse
- Bei Einsatz in der natuerlichen Umgebung ist eine Bildbearbeitung sicherlich erforderlich
'''
class ColorFilter:
	def __init__(self):
		print ("hier ist der Farb-Objekt-Detektor ")
		self.iSync=0     # Index zur Synchronisation zwischen den Nods
		self.frameObjImage=''
		self.allObj = Objekt()
	'''
	Informationen von Prediction werden empfangen und ausgewertet.
	Trash-Objekte werden markiert im frameObjImage und spaeter ausgegeben, in der Methode inFrame'''
	def setPredictionStr(self, predictionStr):
		predictionNumber, probability, comment=predictionStr.data.split("|") # zerlege String
		if comment == "TRASH":
			cv2.line(self.frameObjImage,(self.allObj.x0[self.iSync],self.allObj.y0[self.iSync]),(self.allObj.x1[self.iSync],self.allObj.y1[self.iSync]),(0,0,255),3)
			cv2.line(self.frameObjImage,(self.allObj.x0[self.iSync],self.allObj.y1[self.iSync]),(self.allObj.x1[self.iSync],self.allObj.y0[self.iSync]),(0,0,255),3)
			cv2.putText(self.frameObjImage,comment,(self.allObj.x0[self.iSync],self.allObj.y0[self.iSync]-3),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
		if VERBOSEprio>2: 
			print(predictionNumber, probability, comment, self.iSync) # test
			print("%d,%d,%d, %d"% (self.allObj.x0[self.iSync],  self.allObj.y0[self.iSync] ,self.allObj.x1[self.iSync],self.allObj.y1[self.iSync]))
		self.iSync+=1		
		
	'''Strassenbild wird mehrfach durchlaufen, da Schilder eines Typs auch mehrfach vorkommen koennen.
	Wenn kein neues Farbobjekt erkannt wird, wird die Methode beendet
	analysebild - Strassenszene
	return: frameObjektImage - Strassenszene mit Umrandeten Objekten
	return: allObjImages     - Liste mit gefundenen Objekten  '''
	''' - Laedt eine Strassenszene als Bild und schickt dieses gleich weiter '''
	def inImages(self, analysebildPfad="objDetect/street/mitKreisverkehr.png"):
		image=cv2.imread(analysebildPfad, 1)  # im bereinigten Bild wird gesucht und ggf. gefundene Objekte entfernt
		return self.inFrame(image, bezeichnung="objDetect/street/mitKreisverkehr.png")
	''' - WebCam liefert das Bild direkt '''
	def inFrame(self, image, bezeichnung="From WebCam"): # 
		#Trash-Anzeige im Strassenbild (das ist etwas trickreich, da Ruckkoplung notwendig)
		# siehe Methode setPredictionStr() und Instanzvariablen im Konstruktor
		try:	
			cv2.imshow("street after prediction with TRASH", self.frameObjImage)
			cv2.waitKey(3000) #Pause fur Trashanzeige
		except:
			pass
		#sizeRoadPicture=(1280,720)
		sizeRoadPicture=(800,450)
		image=cv2.resize(image,sizeRoadPicture)  ## Standardgroesse herstellen
		filledObjImage=image.copy()
		self.frameObjImage=image.copy()
		# Suchzyklen fuer mehrere Objekte gleichen Farbtyps
		if VERBOSEprio>2: print("Neues Packet")
		cv2.destroyAllWindows()
		self.allObj = Objekt() ## Neue Liste fuer neue Bildergruppe##
		self.iSync=0
		suchZyklus=0
		while True:
			self.frameObjImage, filledObjImage, obj=self.detectContur(image,self.frameObjImage,filledObjImage) 
			print("Such-Zyklus: %d, Einzelbilder: %d, Name: %s" % (suchZyklus, len(obj.images), bezeichnung))
			# Fuer jeden Suchzyklus Subbilder anhaengen
			for i in range(0,len(obj.images)):
				self.allObj.images.append(obj.images[i])
				self.allObj.x0.append(obj.x0[i])
				self.allObj.y0.append(obj.y0[i])
				self.allObj.x1.append(obj.x1[i])
				self.allObj.y1.append(obj.y1[i])
			suchZyklus=suchZyklus + 1
			if len(obj.images)<=0: break
		cv2.imshow("Strassenbild mit Objektrahmen: "+bezeichnung, self.frameObjImage)	
		cv2.waitKey(500) #Pause fuer Bildanzeige
		return self.allObj.images, self.frameObjImage
	
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
		obj = Objekt()
		# Farbgrenzwerte  b, g, r
		#      Verbote (rot), Hauptstr. (gelb), Gebote (blau) 
		upLim =[[50, 50, 255], [50, 255, 255], [ 255, 50, 50]]
		lowLim=[[ 0,  0,  60], [ 0, 80,   80], [  70,  0,  0]]
		## Gedanke: Es waere denkbar, die Farbgrenzwerte so zu variieren (Zyklus), dass alle moeglichen Objekte
		##          erkannt herangezogen werden. Aus diesenr Menge wird dan durch ein weiteres Verfahren (deep learning)
		##  		eine weitere Eingrenzung vorgenmommen
		#obj.images=[]
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
				# Zusaetzlicher Rand um das Schild/Objekt
				kRand=0.15
				if (i==1): kRand=0.5  # gelbes Schild auf weissem Grund -> Rahmen vergroessert
				xr=int(w * kRand) # Zusaetzlicher Rand 
				yr=int(h * kRand)
				#+++++++++++++++++
				x0 = x - xr
				x1 = x + w + xr
				y0 = y - yr
				y1 = y + h + yr
				wr= float(x1 - x0) # mus float
				hr= float(y1 - y0)
				## Uebernahme der Bereiche, ausschluss unsinniger Bereiche (zu klein, zu schmal)
				if VERBOSEprio>3: print(" Try in ColorZyklus 0,1,2 (x0, y0, wr, hr)", (x0, y0, wr, hr))
				if (wr>31 and hr>31 and (hr/wr)>0.5 and (wr/hr)>0.5):
					cv2.waitKey(100) # notwendig zur Synchronisation der Datenstroeme
					objImg=image[y0:y1,x0:x1] # Objekt-Ausschnitt entnehmen
					obj.images.append(objImg)	# Objekt-Ausschnitt der Liste hinzufuegen
					cv2.rectangle(frameObjImage,(x0,y0),(x1,y1),(upLim[i]),2)
					cv2.rectangle(frameObjImage,(x0-2,y0-2),(x1+2,y1+2),(lowLim[i]),2)
					cv2.rectangle(filledObjImage,(x0,y0),(x1,y1),(255,255,255),cv2.FILLED)
					if VERBOSEprio>9: cv2.imshow("subimage: ", objImg)
					if VERBOSEprio>3: print("ColorZyklus 0,1,2 - Grenzwertabfrage")
					obj.x0.append(x0)
					obj.y0.append(y0)
					obj.x1.append(x1)
					obj.y1.append(y1)
					if VERBOSEprio>3: print("%d,%d,%d, %d"% (obj.y0,obj.y1,obj.x0,obj.x1))
					#+++++++++++++++++++++++++++++++++++++++++++++++++
			except:
				if VERBOSEprio>1: print("Kein Objekt")
				break
		if VERBOSEprio>1: print(" detectContur ende")
		return frameObjImage, filledObjImage, obj     
		

			
		
