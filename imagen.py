from PIL import Image
import os
import numpy as np

def entrenarImagenes(ruta, universidad):
    contador = 0
    cosas = []
    for file in os.listdir(ruta):
        if file.endswith(".jpg"):
            im = Image.open(ruta + "/" + file)
            contador = contador + 1
            #print(im.size)
            #imagen_rgb = img.convert("RGB")
            pixel_values = list(im.getdata())
            pixel_values = np.array(pixel_values)
            cosas.append(pixel_values)
            
            
    #print(pixel_values)
    res = np.array(cosas)
    np.random.shuffle(res)
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(contador*0.7)

    train = res[0:slice_point]
    train_set_x_orig = train.reshape(slice_point,128,128,3)

    test = res[slice_point:contador]
    test_set_x_orig = test.reshape(contador  - slice_point,128,128,3)

    train_set_y_orig = np.array(universidad)
    train_set_y_orig = np.tile(train_set_y_orig,slice_point)

    test_set_y_orig = np.array(universidad)
    test_set_y_orig = np.tile(test_set_y_orig, contador - slice_point)

    print(train_set_x_orig.shape)
    print(test_set_x_orig.shape)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No ' + universidad, universidad]
    

entrenarImagenes("datasets/USAC", "usac")
