#!/usr/bin/env python
# The German Traffic Sign Recognition Benchmark
# http://benchmark.ini.rub.de/
from __future__ import absolute_import #++
from __future__ import division		#++
from __future__ import print_function
import cv2
import rospy
import tensorflow as tf
import keras
import numpy as np
import Image
from keras import backend as k
from keras import callbacks
from keras.datasets import mnist
from keras.datasets import cifar10 # extra
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json ## https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from keras.utils import np_utils
from keras.utils import plot_model
from keras import optimizers
from keras.constraints import maxnorm ## extend
from keras.optimizers import SGD      ## extend
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from PIL import Image  ###
from scipy import ndimage
from sklearn.model_selection import train_test_split ### $ sudo pip install -U scikit-learn
import matplotlib.pyplot as plt ###
import csv ###
import glob, os  ## img convert


k.set_image_dim_ordering( 'th' )
# params for all classes
batch_size =256  ## 128
num_classes = 44  # 43 only signs #44 mit trash #49 without signs
epochs = 44 ## 10 # 12 # 25  ## fuer Test Wert reduziert
lrate = 0.01
verbose_train = 1 # 2
verbose_eval = 0
''' Zwei Modelle stehen zur Verfuegung '''
MODEL_NAME="lcnnModel" #["scnnModel", "lcnnModel"] #  Auswahl ist zu setzen

''' Simple and Large Convolutional Neural Network CNN
    Farbbilder mit shape 32 x 32 x 3 pixel '''   		
class Gtsrb:
	''' Konstruktor prueft ob Modell bereits treniert '''
	def __init__(self, OBJ_ROWS, OBJ_COLS):
		print("class GtsrbCnn")
		global img_rows
		img_rows=OBJ_ROWS
		global img_cols
		img_cols=OBJ_COLS
		self.model = Sequential()
		try:
			self.loadModel(MODEL_NAME)
		except:
			print("-> Modelltraining wird durchgefuehrt!")
			self.modified()
	
	''' function for reading the images
	Reads train traffic sign data for German Traffic Sign Recognition Benchmark.
	Arguments: path to the traffic sign data, for example './GTSRB/Training'
	Returns:   list of images, list of corresponding labels'''
	'''siehe auch: http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset
		       https://matplotlib.org/users/installing.html'''
	''' Modifiziert: J. H 12.2018
	@param rootpath - Pfad fuer die Trainingsbilder
	@param subDirNo - Anzahl der Klassen
	@return Trainingsdaten-Paar, Testdaten-Paar (jeweils als Liste)
	'''
	def readTrafficSigns(self, rootpath="./TrainingImages", subDirNo=num_classes):
		npImages = [] # images
		labels = [] # corresponding labels
		size=[img_rows,img_cols]
		print("Daten werden geladen")
		# loop over all num_classes classes
		for c in range(0,subDirNo,1):
			prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
			#t# print("Verzeichnis: ", prefix)
			gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
			gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
			gtReader.next() # skip header
			# loop over all images in current annotations file
			for row in gtReader:
				ppmImage=Image.open(prefix + row[0])
				ppmNormImage=ppmImage.resize(size)
				npImage=img_to_array(ppmNormImage, data_format = "channels_last")
				# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
				b, g, r = cv2.split(npImage)
				npImage=cv2.merge((r,g,b))
				npImages.append(npImage)
				labels.append(row[7]) # the 8th column is the label
			gtFile.close()
		print("Number of Images: %d" % len(npImages)) 
		## print("image[0]: ", npImages[0]) ## for Test
		# Split das Dataset
		X_train, X_test, y_train, y_test= train_test_split (npImages, labels, test_size=0.1, random_state=33)
		print("Train set size: {0}, Test set size: {1}". format(len(X_train), len(X_test)))
		return (X_train, y_train), (X_test, y_test)	

	'''## Load Data and normalize this  J.Heinke
	@return Trainingsdaten-Paar, Testdaten-Paar (jeweils als Liste)
	'''
	def loadData(self):
		seed = 7 		# fix random seed for reproducibility
		np.random.seed(seed)
		global X_test
		global y_test
		global X_train ##
		global y_train  ##
		## Laden der Bilder mit Labels und aufteilen in Trainings- und Testmenge 		
		(X_train, y_train), (X_test, y_test) = self.readTrafficSigns(rootpath="./TrainingImages", subDirNo=num_classes)
		## images to numpy-array (ist eigentlich schon)
		X_train= np.array(X_train, dtype='float32')
		X_test = np.array(X_test, dtype='float32')
		# reshape to be [samples][width][height][channels]
		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255.0
		X_test  = X_test  / 255.0
		#t# print(X_train[7]) ## Testausgabe
		#t# print("\n   y_test: ", y_test[7]) ## 		
		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test  = np_utils.to_categorical(y_test)
		#-# print("  y_train-Matrix: ", y_train[6]) ##   
		#-# print("  y_test-Matrix: ", y_test[6]) ## 
		return (X_train, y_train), (X_test, y_test)

	''' 2) Define simple cnn Model Quelle: Develop Deep learning ..., Brownlee
	@param subDirNo - Anzahl der Klassen
	@return - Trainingsmodell
	'''
	def scnnModel(self, num_classes):
		self.model.add(Conv2D((32), (3, 3), input_shape=(img_rows, img_cols,3),activation='relu', kernel_constraint=maxnorm(max_value=3)))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols,3), activation='relu', padding='same', kernel_constraint=maxnorm(3))) ##
		self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
		self.model.add(Flatten())
		self.model.add(Dense(units=batch_size, activation='relu', kernel_constraint=maxnorm(3)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(units=num_classes, activation='softmax'))
		return self.model
		
	''' 2to) Define large cnn Model, Quelle: Brownlww
	@param subDirNo - Anzahl der Klassen
	@return - Trainingsmodell
	'''
	def lcnnModel(self, num_classes):
		self.model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols,3), activation= 'relu' , padding= 'same' ))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(32, (3, 3), activation= 'relu' , padding= 'same' ))
		self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
		self.model.add(Conv2D(64, (3, 3), activation= 'relu' , padding= 'same' ))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(64, (3, 3), activation= 'relu' , padding= 'same' ))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Conv2D(128, (3, 3), activation= 'relu' , padding= 'same' ))
		self.model.add(Dropout(0.2))
		self.model.add(Conv2D(128, (3, 3), activation= 'relu' , padding= 'same' ))
		self.model.add(MaxPooling2D(pool_size=(2, 2),  padding='same'))
		self.model.add(Flatten())
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1024, activation= 'relu' , kernel_constraint=maxnorm(3)))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(512, activation= 'relu' , kernel_constraint=maxnorm(3)))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(num_classes, activation= 'softmax' ))
		return self.model
	
	''' Speichert die Modellparameter und das Modell
	--- https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	@param fileName - Pfad, Dateiname, mit dem das Modell gespeichert wird
	'''
	def saveModel(self, fileName="cnnGtsrbModel"):
		print("Saved model.h5 to disk")
		### serialize model to JSON
		model_json = self.model.to_json()
		with open(fileName+".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights(fileName+".h5")
		print("----------------------")
		
	''' Laedt Modell (*.json, *.h5) 
	https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	@param fileName - Pfad, Dateiname, mit dem das Modell geladen wird
	'''
	def loadModel(self, fileName="cnnGtsrbModel"):
		print("Loaded model from disk")
		json_file = open(fileName+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		# load weights into new model
		self.model.load_weights(fileName+".h5")
		return self.model
		
	''' Simple Convolutional Neural Network Training  Modifiziert J.H'''
	def modified(self): 
		## 1a) Load Data ##
		(X_train, y_train), (X_test, y_test)=self.loadData()
		num_classes = y_test.shape[1]
		## 2) Define Baseline Model # build the model
		#v# self.model = self.scnnModel(num_classes) # simpel model
		# # Anweisung wird zusammengestellt, je nach ausgewaehltem Modell
		modelName="self.model=self."+MODEL_NAME+"(num_classes)" 
		exec(modelName) 
		# 3) Compile model
		decay = lrate/epochs
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		self.model.summary()
		## 4) Fit Model
		history=self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
		##history=self.model.fit(X_train, y_train, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=0)		
		## 5) Evaluate Modell ##
		scores = self.model.evaluate(X_test, y_test, verbose=verbose_eval)
		## 6) Speichert Modell 
		self.saveModel(fileName=MODEL_NAME)
		## ----------------------------------------
		print(" Error(class Gtsrb, cnnModel)  : %.2f%%" % (100-scores[1]*100))
		print(" Test accuracy            : %.2f%%" % (scores[1]*100,))
		self.plotHistory(history) # Methode zur Visualisierung und zur Trainings-History
		print()
	
	''' Modelvisualisierung und Trainings-History
	https://keras.io/visualization/#training-history-visualization
	https://keras.rstudio.com/articles/training_visualization.html
	@param history - Abbild des Trainingsvorgangs
	@param pfad - Speicherort der Grafiken
	'''
	def plotHistory(self,history, pfad="./Plot/"):
		# Visualisierung des Modells: https://keras.io/visualization/
		plot_model(self.model, to_file=pfad+MODEL_NAME+'.png',show_shapes=True, show_layer_names=True)
		#----------------------------------------
		# Display Deep Learning Model Training History in Keras
		# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
		# list all data in history
		print(history.history.keys())
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'], label='Test')
		plt.title('model accuracy - '+MODEL_NAME)
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['training', 'validation'],loc='upper left')
		plt.savefig(pfad+MODEL_NAME+'_accu.png')
		# plt.show()
		plt.close()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss - '+ MODEL_NAME)
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training', 'validation'], loc='upper right')
		plt.savefig(pfad+MODEL_NAME+'_loss.png')
		#plt.show()
		plt.close()

	
	''' Bewertet ein einzelnes Bild aus der Testmenge, Quelle: Vorlesungsskript
	@param index - Indes des Testbildes
	@return - Klassennummer des Testbildes, Prognosewert
	'''
	def predictTestImage(self, index=7):
		#t# print("--- print in predictionTestImage() ---")
		## 1a) Load Data, da ggf. noch nicht geladen ##
		## (X_train, y_train), (X_test, y_test)=self.loadData()
		(_, _), (X_test, y_test)=self.loadData()
		# expand dimension for batch
		input_data = np.expand_dims(X_test[index], axis=0)  # tensorflow
		input_data = np.asarray(input_data).astype(np.float32) / 255.0 # j.j
		input_label = y_test[index]
		prediction = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(prediction, axis=None, out=None)
		input_label = np.argmax(input_label, axis=None, out=None)
		# output
		#print("index of the picture: %s" % (index,))
		#print("prediction label    : %s" % (prediction,))
		#print("real label          : %s" % (input_label,))
		return input_label, prediction
		
	''' Bewertet ein einzelnes Bild, Modifiziert J.H 
	@param - input_data - Bild fuer das eine Prognose vorgenommen wird
	@return - Prognosewert und Wahrscheinlichkeitswert fuer die vorgenommen Prognose
	'''
	def predictImage(self, input_data):
		input_data = np.expand_dims(input_data, axis=0)  # tensorflow + 1 dimension
		input_data = np.asarray(input_data).astype(np.float32) / 255.0 # j.j
		predictions = self.model.predict(input_data)
		# revert from one-hot encoding
		prediction = np.argmax(predictions,axis=None, out=None)
		# Wahrscheinlichkeiten fuer die einzelnen Klassen
		probabilities=self.model.predict(input_data.reshape(1,img_rows, img_cols,3))
		#t# print ("probabilities of prediction:")
		#t# print(probabilities)
		probability=np.amax(probabilities, axis=None, out=None) # Maximalwert, Axsenunabhaengig
		probabilitySort=np.sort(probabilities, axis=None)[::-1] # absteigend sortieren
		#t# print("probabilities Sort: ")
		#t# print(probabilitySort)
		# output
		#+# print("--- print in predictionImage() ---")
		print("prediction label    : %s with the probability %-10.8f" % (prediction, probability,))
		print("second probability  : ", probabilitySort[1])
		print("third probability   : ", probabilitySort[2])
		print("===========================================")
		return  prediction, probability	
		

	
		
			

