#!/usr/bin/env python
# -*- coding: utf-8 -*-

from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np
import os
from PIL import Image

ONLY_SHOW = False #Veo si quiero mostrar una imagen del conjunto de datos

#Cargando conjuntos de datos
#usac
train_set_x_orig_usac, train_set_y_usac, test_set_x_orig_usac, test_set_y_usac, classes_usac = File.entrenarImagenesUsac()
#mariano
train_set_x_orig_mari, train_set_y_mari, test_set_x_orig_mari, test_set_y_mari, classes_mari = File.entrenarImagenesMari()
#landivar
train_set_x_orig_land, train_set_y_land, test_set_x_orig_land, test_set_y_land, classes_land = File.entrenarImagenesLand()
#marroquin
train_set_x_orig_marro, train_set_y_marro, test_set_x_orig_marro, test_set_y_marro, classes_marro = File.entrenarImagenesMarro()

if ONLY_SHOW:
    index = 14 #Gato
    #index = 100 #No Gato
    #index = 59 #Gato
    Plotter.show_picture(train_set_x_orig_mari[index])
    #print(classes[train_set_y_mari[0][index]])
    exit()

#Usac
# Convertir imagenes a un solo arreglo
train_set_x_usac = train_set_x_orig_usac.reshape(train_set_x_orig_usac.shape[0], -1).T
test_set_x_usac = test_set_x_orig_usac.reshape(test_set_x_orig_usac.shape[0], -1).T
# Definir los conjuntos de datos
train_set_usac = Data(train_set_x_usac, train_set_y_usac, 255)
test_set_usac = Data(test_set_x_usac, test_set_y_usac, 255)

#Mariano
#train_set_x_mari = train_set_x_orig_mari.reshape(train_set_x_orig_mari.shape[0], -1).T
#test_set_x_mari = test_set_x_orig_mari.reshape(test_set_x_orig_mari.shape[0], -1).T
# Definir los conjuntos de datos
#train_set_mari = Data(train_set_x_mari, train_set_y_mari, 255)
#test_set_mari = Data(test_set_x_mari, test_set_y_mari, 255)

#Landivar
#train_set_x_land = train_set_x_orig_land .reshape(train_set_x_orig_land.shape[0], -1).T
#test_set_x_land = test_set_x_orig_land .reshape(test_set_x_orig_land.shape[0], -1).T
# Definir los conjuntos de datos
#train_set_land = Data(train_set_x_land, train_set_y_land, 255)
#test_set_land = Data(test_set_x_land, test_set_y_land, 255)

#Marroquin
#train_set_x_marro = train_set_x_orig_marro.reshape(train_set_x_orig_marro.shape[0], -1).T
#test_set_x_marro = test_set_x_orig_marro.reshape(test_set_x_orig_marro.shape[0], -1).T
# Definir los conjuntos de datos
#train_set_marro = Data(train_set_x_marro, train_set_y_marro, 255)
#test_set_marro = Data(test_set_x_marro, test_set_y_marro, 255)

print("Usac: ")
print('Original: ', train_set_x_orig_usac.shape)
print('Con reshape: ', train_set_x_usac.shape)

print("Landivar: ")
#print('Original: ', train_set_x_orig_land.shape)
#print('Con reshape: ', train_set_x_land.shape)

print("Marroquin: ")
#print('Original: ', train_set_x_orig_marro.shape)
#print('Con reshape: ', train_set_x_marro.shape)

print("Mariano: ")
#print('Original: ', train_set_x_orig_mari.shape)
#print('Con reshape: ', train_set_x_mari.shape)
#exit()

# Se entrenan los modelos
modelUsac = Model(train_set_usac, test_set_usac, reg=False, alpha=0.001, lam=0)
#modelUsac2 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.0001, lam=0) 
#modelUsac3 = Model(train_set_usac, test_set_usac, reg=False, alpha=0.01, lam=100)
  
#modelMari = Model(train_set_mari, test_set_mari, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización

#modelLand = Model(train_set_land, test_set_land, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización

#modelMarro = Model(train_set_marro, test_set_marro, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización

modelUsac.training()
#modelUsac2.training()
#modelUsac3.training()
#modelMari.training()
#modelLand.training()
#modelMarro.training()


#model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0)
#model1.training()

#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=1) #Se puede ver en la gráfica que hay SOBRE-AJUSTE
#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=150) #Aquí también se puede ver sobre-ajuste

#model2 = Model(train_set, test_set, reg=True, alpha=0.001, lam=300) #Se ajusta mejor con la regulariación de 300, pero se tarda más

# Se grafican los entrenamientos
#Plotter.show_Model([model1, model2])
#Plotter.show_Model([model1])
Plotter.show_Model([modelUsac])

loop = True
while loop:
    cosas = [1]
    ruta = input("Ruta de carpeta de imagenes: ")
    for file in os.listdir(ruta):
        if file.endswith(".jpg"):
            im = Image.open(ruta + "/" + file)
            print(ruta + "/" + file)
            pixel_values = list(im.getdata())
            pixel_values = np.array(pixel_values).reshape(1,128,128,3)
            res = pixel_values.reshape(pixel_values.shape[0], -1).T
            res = np.insert(res, 0, 1, axis=0)
            print(modelUsac.predict(res))
