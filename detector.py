import tensorflow as tf
from keras import layers, models
import os
import numpy as np
import cv2
import random

width = 300
height = 300
ruta_train = 'proyectoPY/PY/datos/train/'
ruta_predict = 'proyectoPY/datos/carros/valid/images/-1AED8217-3B98-4458-83DD-32F5CA3851FF-png_jpg.rf.227b47cf5e0a610fb1d4443bbcfa3afb.jpg'

train_x = []
train_y = []

labels = os.listdir(ruta_train)

for i in os.listdir(ruta_train):
    for j in os.listdir(ruta_train + i):
        img = cv2.imread(ruta_train + i + '/' + j)
        if img is not None:
            resized_image = cv2.resize(img, (width, height))
            train_x.append(resized_image)
            for x, y in enumerate(labels):
                if y == i:
                    array = np.zeros(len(labels))
                    array[x] = 1
                    train_y.append(array)
        else:
            print(f"no pude leer la imagen {ruta_train + i + '/' + j}")

x_data = np.array(train_x)
y_data = np.array(train_y)

#Comentar la parte del entrenamiento
#model = tf.keras.Sequential([
     #layers.Conv2D(32, (3, 3), input_shape=(width, height, 3)),
     #layers.Activation('relu'),
     #layers.MaxPooling2D(pool_size=(2, 2)),
     #layers.Conv2D(32, (3, 3)),
     #layers.Activation('relu'),
     #layers.MaxPooling2D(pool_size=(2, 2)),
     #layers.Conv2D(64, (3, 3)),
     #layers.Activation('relu'),
    #layers.MaxPooling2D(pool_size=(2, 2)),
    #layers.Flatten(),
    #layers.Dense(64),
    # layers.Activation('relu'),
   #  layers.Dropout(0.5),
  #  layers.Dense(len(labels)),
 #    layers.Activation('sigmoid')
#])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#epochs = 100

#model.fit(x_data, y_data, epochs=epochs)

#models.save_model(model, 'mimodelo.keras')

# Cargar el modelo guardado
model = models.load_model('proyectoPY/mimodelo.keras')

# Ejecutar la predicción
my_image = cv2.imread(ruta_predict)

if my_image is not None:
    my_image = cv2.resize(my_image, (width, height))
    result = model.predict(np.array([my_image]))[0]

    porcentaje = max(result) * 100
    grupo = labels[result.argmax()]
    #gp = grupo
    
    print(grupo, round(porcentaje),"%")
    
    #*////////////////////////////////////LECTURA D//////////////////////////////////////////////
    from gtts import gTTS
    import pygame

# Crear y escribir en lectura.txt
    with open("proyectoPY/lectura.txt", "w") as file:
        file.write(grupo)

# Definir la función voz
    def voz(text_file, lang, name_file):
        with open(text_file, "r") as file:
            text = file.read()
        tts = gTTS(text=text, lang=lang)
        tts.save(name_file)

# Usar el nombre de archivo correcto
    voz("proyectoPY/lectura.txt", "es", "voz.mp3")

# Inicializar pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load("voz.mp3")
    pygame.mixer.music.play()

    #print("Reproduciendo")

# Esperar hasta que la reproducción termine
    while pygame.mixer.music.get_busy():
        continue
    #/////////////////////////////////////////////////////////////////////////////////
    
    cv2.imshow('Prediccion', my_image)
    cv2.waitKey(0)
    
else:
    print(f"Error: la imagen no se detecto {ruta_predict}")