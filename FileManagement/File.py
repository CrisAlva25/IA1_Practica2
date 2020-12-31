#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os

source = None

class nodito:
    def __init__(self, arr, val):
        self.arr = arr
        self.val = val


def obtenerImagen(ruta, val):
    contador = 0
    cosas = []

    #unicamente usac
    for file in os.listdir(ruta):
        if file.endswith(".jpg"):
            im = Image.open(ruta + "/" + file)
            contador = contador + 1
            #print(im.size)
            #imagen_rgb = img.convert("RGB")
            pixel_values = list(im.getdata())
            #print(pixel_values)
            cosas.append(nodito(pixel_values,val))
    return contador, cosas

def entrenarImagenesUsac():
    total = 0
    imgs = []

    contA, imgAux = obtenerImagen("datasets/USAC",1)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Mariano",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Landivar",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Marroquin",0)
    total = total + contA
    imgs = imgs + imgAux

    #print(pixel_values)
    res = np.array(imgs)
    np.random.shuffle(res)
    tx = []
    ty = []
    for element in res:
        tx.append(element.arr)
        ty.append(element.val)
    train_x = np.array(tx).reshape(total,128,128,3)
    train_y = np.array(ty).reshape(total,1)
    #print(res)
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(total*0.7)

    train_set_x_orig = train_x[:slice_point]
    test_set_x_orig = train_x[slice_point:]

    train_set_y_orig = train_y[:slice_point]
    test_set_y_orig = train_y[slice_point:]

    #print(train_set_x_orig.shape)
    #print(test_set_x_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No usac', 'usac']

def entrenarImagenesLand():
    total = 0
    imgs = []

    contA, imgAux = obtenerImagen("datasets/USAC",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Mariano",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Landivar",1)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Marroquin",0)
    total = total + contA
    imgs = imgs + imgAux

    #print(pixel_values)
    res = np.array(imgs)
    np.random.shuffle(res)
    tx = []
    ty = []
    for element in res:
        tx.append(element.arr)
        ty.append(element.val)
    train_x = np.array(tx).reshape(total,128,128,3)
    train_y = np.array(ty).reshape(total,1)
    #print(res)
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(total*0.7)

    train_set_x_orig = train_x[:slice_point]
    test_set_x_orig = train_x[slice_point:]

    train_set_y_orig = train_y[:slice_point]
    test_set_y_orig = train_y[slice_point:]

    #print(train_set_x_orig.shape)
    #print(test_set_x_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No landivar', 'landivar']

def entrenarImagenesMari():
    total = 0
    imgs = []

    contA, imgAux = obtenerImagen("datasets/USAC",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Mariano",1)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Landivar",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Marroquin",0)
    total = total + contA
    imgs = imgs + imgAux

    #print(pixel_values)
    res = np.array(imgs)
    np.random.shuffle(res)
    tx = []
    ty = []
    for element in res:
        tx.append(element.arr)
        ty.append(element.val)
    train_x = np.array(tx).reshape(total,128,128,3)
    train_y = np.array(ty).reshape(total,1)
    #print(res)
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(total*0.7)

    train_set_x_orig = train_x[:slice_point]
    test_set_x_orig = train_x[slice_point:]

    train_set_y_orig = train_y[:slice_point]
    test_set_y_orig = train_y[slice_point:]

    #print(train_set_x_orig.shape)
    #print(test_set_x_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No mariano', 'mariano']

def entrenarImagenesMarro():
    total = 0
    imgs = []

    contA, imgAux = obtenerImagen("datasets/USAC",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Mariano",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Landivar",0)
    total = total + contA
    imgs = imgs + imgAux

    contA, imgAux = obtenerImagen("datasets/Marroquin",1)
    total = total + contA
    imgs = imgs + imgAux

    #print(pixel_values)
    res = np.array(imgs)
    np.random.shuffle(res)
    tx = []
    ty = []
    for element in res:
        tx.append(element.arr)
        ty.append(element.val)
    train_x = np.array(tx).reshape(total,128,128,3)
    train_y = np.array(ty).reshape(total,1)
    #print(res)
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(total*0.7)

    train_set_x_orig = train_x[:slice_point]
    test_set_x_orig = train_x[slice_point:]

    train_set_y_orig = train_y[:slice_point]
    test_set_y_orig = train_y[slice_point:]

    #print(train_set_x_orig.shape)
    #print(test_set_x_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No marroquin', 'marroquin']

