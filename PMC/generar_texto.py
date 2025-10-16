import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funciones_auxiliares import cargar_corpus, cargar_modelo

corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus("corpus.txt", "corpus")

W1, W2, N, C, eta = cargar_modelo("pesos_cbow_pcshavak-c_epoca1600.npz", "relevant_weights")

modelo = keras.models.load_model("/home/franco/Escritorio/Repos/aprendizaje-automatico-avanzado/PMC/modelo_pmc.keras")

def predecir_palabra_siguiente(modelo, lista_palabras, W1, word_to_idx, idx_to_word):
    
    x = []
    #secuencia = input(f"Ingrese una secuencia de longitud {ventana}: ")
    #lista_palabras = np.array(secuencia.split())
    idx_secuencia = [word_to_idx[palabra] for palabra in lista_palabras]
    x.append(W1[idx_secuencia].flatten().reshape(1, -1))
    prediccion = modelo.predict(x)
    indice_pred = prediccion.argmax(axis=1)[0]
    palabra_pred = idx_to_word[indice_pred]
    return palabra_pred
    #print("La palabra siguiente es: ", palabra_pred)
    

#predecir_palabra_siguiente(modelo, W1, word_to_idx, idx_to_word)

def generar_texto(modelo, word_to_idx, idx_to_word, longitud, ventana):
    
    secuencia = input(f"Ingrese una secuencia de longitud {ventana}: ")
    lista_palabras = np.array(secuencia.split())
    texto = secuencia
    
    while longitud != 0:
        prediccion = predecir_palabra_siguiente(modelo, lista_palabras, W1, word_to_idx, idx_to_word)
        texto += f" {prediccion}"
        lista_palabras = np.append(lista_palabras[1:], prediccion)
        longitud -= 1
    print(texto)

generar_texto(modelo, word_to_idx, idx_to_word, 30, 5) #El corpus guardo algunos indices de pagina :'(
        
        
        
    