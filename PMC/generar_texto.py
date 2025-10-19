import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funciones_auxiliares import cargar_corpus, cargar_modelo

corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus("corpus.txt", "corpus")

W1, W2, N, C, eta = cargar_modelo("pesos_cbow_pcshavak-c_epoca1600.npz", "relevant_weights")

modelo = keras.models.load_model("/home/franco/Escritorio/Repos/aprendizaje-automatico-avanzado/PMC/modelo_pmc2.keras")

def normalizar_vectores(w1):
    normas = np.linalg.norm(w1, axis=1)
    return w1 / normas[:, np.newaxis]

import numpy as np

def normalizar_vectores(matriz):
    normas = np.linalg.norm(matriz, axis=1)
    normas[normas == 0] = 1 
    return matriz / normas[:, np.newaxis]


def predecir_palabras_siguientes_sd(modelo, lista_palabras, W1, word_to_idx, idx_to_word):
    
    idx_secuencia = [word_to_idx[palabra] for palabra in lista_palabras]
    x_input = W1[idx_secuencia]
    x_input_flat = x_input.flatten().reshape(1, -1) 
    prediccion = modelo.predict(x_input_flat)
    prediccion_norm = prediccion / np.linalg.norm(prediccion)
    w1_norm = normalizar_vectores(W1)
    
    similaridades = np.dot(w1_norm, prediccion_norm.T)
    
    sim_ordenadas = np.argsort(similaridades.flatten())[::-1]
    indices_mas_similares = sim_ordenadas[:5]
    palabras_mas_probables = [idx_to_word[i] for i in indices_mas_similares]
    
    return palabras_mas_probables


def predecir_palabras_siguientes_sm(modelo, lista_palabras, W1, word_to_idx, idx_to_word):
    
    idx_secuencia = [word_to_idx[palabra] for palabra in lista_palabras]
    x_input = W1[idx_secuencia]
    prediccion = modelo.predict(x_input, verbose=0)[0]
    indices_mas_probables = np.argsort(prediccion)[::-1][:5]  
    palabras_mas_probables = [idx_to_word[i] for i in indices_mas_probables]
    
    return palabras_mas_probables

def generar_texto(modelo, tipo_salida, word_to_idx, idx_to_word, longitud, ventana):
    
    secuencia = input(f"Ingrese una secuencia de longitud {ventana}: ")
    lista_palabras = np.array(secuencia.split())
    texto = secuencia
    
    while longitud != 0:
        ultimas_tres = lista_palabras[-3:]
        
        if tipo_salida == 'sd':
            prediccion = predecir_palabras_siguientes_sd(modelo, lista_palabras, W1, word_to_idx, idx_to_word)
        else:
            prediccion = predecir_palabras_siguientes_sm(modelo, lista_palabras, W1, word_to_idx, idx_to_word)

        for i in range(len(prediccion)):
            palabra_candidata = prediccion[i]
            if palabra_candidata not in ultimas_tres:
                texto += f" {palabra_candidata}"
                lista_palabras = np.append(lista_palabras[1:], palabra_candidata)
                break
        
        longitud -= 1
    
    print(texto)

generar_texto(modelo, 'sm' word_to_idx, idx_to_word, 30, 8)