import numpy as np
#import cupy as np
import os
import random
import re

# Funciones auxiliares, ideal UNIFICARLAS en algún momento

# ==============================================================
#                      FUNCIONES MATÍAS
# ==============================================================

def cargar_corpus(nombre_archivo="corpus.txt", carpeta="corpus"):
    try:
        ruta = os.path.join(carpeta, nombre_archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
        
        vocab = sorted(set(corpus))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}

        print("Tamaño de corpus:", len(corpus))
        print("Tamaño de vocabulario:", vocab_size)

        return corpus, vocab, vocab_size, word_to_idx, idx_to_word

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}' en '{carpeta}'.")
        return None, None, None, None, None

def inicializar_pesos(vocab_size, N, W1= None, W2=None, nparray=False):
    
    libreria = np if nparray else np
    if W1 is None or W2 is None:
        W1 = libreria.random.normal(0, 0.1, (vocab_size, N))
        W2 = libreria.random.normal(0, 0.1, (N, vocab_size))
    elif nparray:
        W1 = np.asarray(W1)
        W2 = np.asarray(W2)
    
    return W1, W2

def softmax_np(u):
    u_max = np.max(u) # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u)

def softmax_np2(u):
    u_max = np.max(u)           # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)   # e^(u - max)
    sum_exp = np.sum(exp_u)     # suma de exponentes
    softmax = exp_u / sum_exp
    return softmax, sum_exp

def softmax_np(u):
    u_max = np.max(u) # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u)

def softmax_np2(u):
    u_max = np.max(u)           # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)   # e^(u - max)
    sum_exp = np.sum(exp_u)     # suma de exponentes
    softmax = exp_u / sum_exp
    return softmax, sum_exp

def sigmoide_np(x):
    return 1 / (1 + np.exp(-x))

def sigmoide_np(x):
    return 1 / (1 + np.exp(-x))

# Guardado y cargado de modelo

def guardar_modelo(nombre_archivo, W1, W2, eta, N, C, nparray=False):
    if nparray:
        W1 = np.asnumpy(W1)
        W2 = np.asnumpy(W2)
    os.makedirs("weights", exist_ok=True)
    ruta_completa = os.path.join("weights", nombre_archivo)

    np.savez(ruta_completa, W1=W1, W2=W2, eta=eta, N=N, C=C)
    print(f"Pesos e hiperparámetros guardados exitosamente en '{ruta_completa}'")

def cargar_modelo(nombre_archivo=str, carpeta="weights"):
    try:
        ruta = os.path.join(carpeta, nombre_archivo)
        data = np.load(ruta)
        
        W1 = data['W1']
        W2 = data['W2']
    
        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()

        return W1, W2, N, C, eta
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}' en '{carpeta}'.")
        return None, None, None, None, None

# Funciones para el diccionario

def generar_tuplas_central_contexto(corpus, word_to_idx, C=4):
    tuplas = []
    for i in range(C, len(corpus)-C):
        # Palabra central
        palabra_central = corpus[i]
        palabra_central_indice = word_to_idx[palabra_central]

        # Palabras de contexto
        palabras_contexto = corpus[i-C:i] + corpus[i+1:i+C+1]
        palabras_contexto_indices = [word_to_idx[word] for word in palabras_contexto]

        tuplas.append([palabra_central_indice, palabras_contexto_indices])
    return tuplas

def generar_tuplas_central_contexto_negativos(corpus, word_to_idx, C=4, K=5):
    tuplas = []
    distancia_max = C + K
    for i in range(distancia_max, len(corpus) - distancia_max):
        # Índice de la palabra central
        indice_central = word_to_idx[corpus[i]]

        # Índices del contexto interno (POSITIVOS)
        indices_interno = [word_to_idx[w] for w in corpus[i - C : i] + corpus[i + 1 : i + C + 1]]
        
        # Índices del contexto externo (NEGATIVOS)
        indices_externo = [word_to_idx[w] for w in corpus[i - distancia_max : i - C] + corpus[i + C + 1 : i + distancia_max + 1]]
        
        tuplas.append((indice_central, indices_interno, indices_externo))
    return tuplas

# Interacción con el modelo

def ver_palabras_similares(corpus, word_to_idx, idx_to_word, palabra, W1, N=5):
    if palabra in corpus:
        i_palabra = word_to_idx[palabra]

        embedding_palabra = W1[i_palabra]

        productos = W1 @ embedding_palabra # (|V|, N) @ (N, 1)
        #print(f"Shape de embedding: {embedding_palabra.shape}, shape de W1: {W1.shape}, shape de productos: {productos.shape}")
        productos[i_palabra] = -np.inf  # para no recomendarse a sí misma
        indices = np.argpartition(productos, -N)[-N:]
        indices_ordenados = indices[np.argsort(productos[indices])[::-1]]
        similares = [idx_to_word[i] for i in indices_ordenados]

        print(f"Palabras similares a '{palabra}': {similares}")
    else:
        print(f"La palabra {palabra} no existe en el corpus")

def evaluar_cbow_np(indice_tuplas, W1, W2, N=5):
    """
    Evalúa un modelo CBOW devolviendo aciertos top-N (flexibles)
    y aciertos estrictos (exactos), mostrando progreso y porcentajes parciales.
    """
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    
    totales = len(indice_tuplas)
    aciertos_topN = 0
    aciertos_estrictos = 0

    for i, (i_central, i_contextos) in enumerate(indice_tuplas):
        # ---Propagación---
        h = np.mean(W1[i_contextos], axis=0).reshape(-1, 1)
        u = W2.T @ h
        y = softmax_np(u).flatten() 

        # ---Top-N---
        top_n_indices = np.argpartition(y, -N)[-N:]
        if np.any(top_n_indices == i_central):
            aciertos_topN += 1
        
        # ---Acierto estricto (top-1)---
        pred_top1 = np.argmax(y)
        if pred_top1 == i_central:
            aciertos_estrictos += 1

        # ---Progreso---
        if i % 1000 == 0 and i > 0:
            progreso = (i / totales) * 100
            porc_topN = (aciertos_topN / i) * 100
            porc_estrictos = (aciertos_estrictos / i) * 100
            print(f"Progreso: [{progreso:.1f}%] - Top-{N}: {porc_topN:.2f}% | Estrictos: {porc_estrictos:.2f}%", end='\r')

    print(f"\nAciertos Top-{N}: {aciertos_topN}/{totales} ({(aciertos_topN / totales) * 100:.2f}%)")
    print(f"Aciertos estrictos: {aciertos_estrictos}/{totales} ({(aciertos_estrictos / totales) * 100:.2f}%)")
    
    return aciertos_topN, aciertos_estrictos, totales

def evaluar_cbow_np(indice_tuplas, W1, W2, N=5):
    """
    Evalúa un modelo CBOW devolviendo aciertos top-N (flexibles)
    y aciertos estrictos (exactos), mostrando progreso y porcentajes parciales.
    """
    totales = len(indice_tuplas)
    aciertos_topN = 0
    aciertos_estrictos = 0

    for i, (i_central, i_contextos) in enumerate(indice_tuplas):
        # ---Propagación---
        h = np.mean(W1[i_contextos], axis=0).reshape(-1, 1)
        u = W2.T @ h
        y = softmax_np(u).flatten() 

        # ---Top-N---
        top_n_indices = np.argpartition(y, -N)[-N:]
        if np.any(top_n_indices == i_central):
            aciertos_topN += 1
        
        # ---Acierto estricto (top-1)---
        pred_top1 = np.argmax(y)
        if pred_top1 == i_central:
            aciertos_estrictos += 1

        # ---Progreso---
        if i % 1000 == 0 and i > 0:
            progreso = (i / totales) * 100
            porc_topN = (aciertos_topN / i) * 100
            porc_estrictos = (aciertos_estrictos / i) * 100
            print(f"Progreso: [{progreso:.1f}%] - Top-{N}: {porc_topN:.2f}% | Estrictos: {porc_estrictos:.2f}%", end='\r')

    print(f"\nAciertos Top-{N}: {aciertos_topN}/{totales} ({(aciertos_topN / totales) * 100:.2f}%)")
    print(f"Aciertos estrictos: {aciertos_estrictos}/{totales} ({(aciertos_estrictos / totales) * 100:.2f}%)")
    
    return aciertos_topN, aciertos_estrictos, totales

def softmax_np_t(u, axis=0):
    u_max = np.max(u, axis=axis, keepdims=True)
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u, axis=axis, keepdims=True)

def evaluar_cbow_lotes(indice_tuplas, W1, W2, N=5, batch_size=1024):
    """
    Evalúa un modelo CBOW procesando los datos en lotes para mayor eficiencia.
    """
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    
    totales = len(indice_tuplas)
    aciertos = 0

    # El bucle for ahora itera sobre los datos en lotes (batches)
    for i in range(0, totales, batch_size):
        lote_actual = indice_tuplas[i : i + batch_size]
        if not lote_actual:
            continue
            
        batch_central_indices, batch_contextos = zip(*lote_actual)
        
        # ---Propagación (para todo el lote)---
        indices_contexto_aplanados = [idx for sublist in batch_contextos for idx in sublist]
        embeddings = W1[indices_contexto_aplanados]
        
        num_palabras_contexto = len(batch_contextos[0])
        embeddings_lote = embeddings.reshape(len(lote_actual), num_palabras_contexto, -1)
        
        h_lote = np.mean(embeddings_lote, axis=1)
        u_lote = W2.T @ h_lote.T
        y_lote = softmax_np_t(u_lote)

        # ---Tomar N mayores y contar aciertos (para todo el lote)---
        top_n_indices_lote = np.argpartition(y_lote, -N, axis=0)[-N:, :]
        aciertos_mask = (top_n_indices_lote == np.array(batch_central_indices))
        aciertos_en_lote = np.any(aciertos_mask, axis=0).sum()
        aciertos += aciertos_en_lote.item()
        
        # Imprimir el progreso periódicamente
        # (i // batch_size) nos da el número de lote actual
        if (i // batch_size) % 10 == 0:
            progreso = (i + len(lote_actual)) / totales * 100
            print(f"Progreso: [{progreso:.1f}%] - Aciertos: {aciertos}", end='\r')

    # Limpiar la línea de progreso e imprimir el resultado final
    print(" " * 80, end='\r') # Limpia la línea de progreso
    print(f"Cantidad de aciertos (Top-{N}): {aciertos}/{totales}, esto es un {(aciertos / totales) * 100:.2f}%")
    
    return aciertos, totales

def embeber_datos(corpus: list, W1: np.ndarray, word_to_idx: dict, ventana: int):
    n = len(corpus)
    x_train = []
    y_train = []

    print("Incio de la creacion de x_train e y_train.")
    
    for i in range(n - ventana): 
        palabras_ventana = np.array(corpus[i : i + ventana])
        idx_ventana = [word_to_idx[palabra] for palabra in palabras_ventana]
        idx_siguiente = word_to_idx[corpus[i + ventana]]

        x_train.append(W1[idx_ventana].flatten())
        y_train.append(W1[idx_siguiente])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("FIn de la creacion de x_train e y_train.")
    
    return x_train, y_train
