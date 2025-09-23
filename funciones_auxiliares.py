import numpy as np
import cupy as cp
import os
import random
import re

# Funciones auxiliares, ideal UNIFICARLAS en algún momento

# ==============================================================
#                      FUNCIONES MATÍAs
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

def inicializar_pesos(vocab_size, N, W1= None, W2=None, cparray=False):
    
    libreria = cp if cparray else np
    if W1 is None or W2 is None:
        W1 = libreria.random.normal(0, 0.1, (vocab_size, N))
        W2 = libreria.random.normal(0, 0.1, (N, vocab_size))
    elif cparray:
        W1 = cp.asarray(W1)
        W2 = cp.asarray(W2)
    
    return W1, W2

def softmax_np(u):
    u_max = np.max(u) # Estabiliza restando el máximo
    exp_u = np.exp(u - u_max)
    return exp_u / np.sum(exp_u)

def softmax_cp(u):
    u_max = cp.max(u) # Estabiliza restando el máximo
    exp_u = cp.exp(u - u_max)
    return exp_u / cp.sum(exp_u)

def sigmoide_np(x):
    return 1 / (1 + np.exp(-x))

def sigmoide_cp(x):
    return 1 / (1 + cp.exp(-x))

# Guardado y cargado de modelo

def guardar_modelo(nombre_archivo, W1, W2, eta, N, C, cparray=False):
    if cparray:
        W1 = cp.asnumpy(W1)
        W2 = cp.asnumpy(W2)
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

# ==============================================================
#                      FUNCIONES NICOLÁS
# ==============================================================

# auxiliares_cbow.py

"""
palabras_a_indice = {}
palabras_a_onehot = {}

with open("corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

for token in words:
    if token not in palabras_a_indice:

        index = len(palabras_a_indice)
        palabras_a_indice[token] = index

cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    palabras_a_onehot[token] = one_hot_vector
"""

def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return e_u / e_u.sum()

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def generar_tuplas(corpus, palabras_a_indice, contexto):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto])
    for i in indices:

        indices_tuplas.append((palabras_a_indice[corpus[i]], [palabras_a_indice[corpus[i+j]] for j in indices_contexto]))

    return indices_tuplas      



def generar_tuplas_con_negativos(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]


    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            obtener_negativas(corpus,i,contexto,num_negativos)))# negativos

    return indices_tuplas

def obtener_negativas(corpus, indice, contexto, num_negativos=5):
    # Hasta dónde mirar (contexto + margen de negativas)
    maximo = contexto + num_negativos // 2

    # Ventana de contexto
    inicio = max(0, indice - maximo)
    fin = min(len(corpus), indice + maximo + 1)

    # Negativas candidatas izquierda y derecha
    izq = corpus[max(0, inicio - maximo):inicio]
    der = corpus[fin:min(len(corpus), fin + maximo)]

    # Balanceo: si falta en izquierda, saco más de derecha (y viceversa)
    if len(izq) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(izq)
        der = corpus[fin:min(len(corpus), fin + maximo + faltan)]

    elif len(der) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(der)
        izq = corpus[max(0, inicio - maximo - faltan):inicio]

    # Filtrar: excluir la palabra central
    candidatas = [p for p in izq + der if p != corpus[indice]]

    # Convertir a índices y devolver solo las necesarias
    return [palabras_a_indice[p] for p in candidatas[:num_negativos]]


def generar_tuplas_con_negativos_random(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    vocabulario = list(set(range(len(palabras_a_indice))))

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            random.sample(vocabulario - set([palabras_a_indice[corpus[i + j]] for j in indices_contexto]), k=num_negativos)))# negativos

    return indices_tuplas

# auxiliares_skipgram.py

"""
palabras_a_indice = {}
palabras_a_onehot = {}

with open("corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

for token in words:
    if token not in palabras_a_indice:

        index = len(palabras_a_indice)
        palabras_a_indice[token] = index

cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    palabras_a_onehot[token] = one_hot_vector
"""

def softmax(u):
    u_max = np.max(u)
    e_u = np.exp(u - u_max)
    return e_u / e_u.sum()

def sigmoid(x):
    return (cp.tanh(x / 2) + 1) / 2

def generar_tuplas(corpus, palabras_a_indice, contexto):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto])
    for i in indices:

        indices_tuplas.append((palabras_a_indice[corpus[i]], [palabras_a_indice[corpus[i+j]] for j in indices_contexto]))

    return indices_tuplas      



def generar_tuplas_con_negativos(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]


    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            obtener_negativas(corpus,i,contexto,num_negativos)))# negativos

    return indices_tuplas

def obtener_negativas(corpus, indice, contexto, num_negativos=5):
    # Hasta dónde mirar (contexto + margen de negativas)
    maximo = contexto + num_negativos // 2

    # Ventana de contexto
    inicio = max(0, indice - maximo)
    fin = min(len(corpus), indice + maximo + 1)

    # Negativas candidatas izquierda y derecha
    izq = corpus[max(0, inicio - maximo):inicio]
    der = corpus[fin:min(len(corpus), fin + maximo)]

    # Balanceo: si falta en izquierda, saco más de derecha (y viceversa)
    if len(izq) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(izq)
        der = corpus[fin:min(len(corpus), fin + maximo + faltan)]

    elif len(der) < num_negativos // 2:
        faltan = (num_negativos // 2) - len(der)
        izq = corpus[max(0, inicio - maximo - faltan):inicio]

    # Filtrar: excluir la palabra central
    candidatas = [p for p in izq + der if p != corpus[indice]]

    # Convertir a índices y devolver solo las necesarias
    return [palabras_a_indice[p] for p in candidatas[:num_negativos]]


def generar_tuplas_con_negativos_random(corpus, palabras_a_indice, contexto, num_negativos):

    ## Genero una lista de todos los indices de las palabras en el corpus, posibles sin padding
    indices = [i for i in range(contexto,(len(corpus)-contexto))]

    ## Genero una lista de los contextos de las palabras de contexto
    indices_contexto = [i for i in range(-contexto,0)] + [i for i in range(1,contexto+1)]
    vocabulario = list(set(range(len(palabras_a_indice))))

    indices_tuplas = []

    ## Por cada indice en indices, genero una tupla (indice_central, [indices_contexto], [indices_negativos])
    for i in indices:

        indices_tuplas.append(
            (palabras_a_indice[corpus[i]],  # target central
            [palabras_a_indice[corpus[i + j]] for j in indices_contexto],  # contexto
            random.sample(vocabulario - set([palabras_a_indice[corpus[i + j]] for j in indices_contexto]), k=num_negativos)))# negativos


    return indices_tuplas

# ==============================================================
#                      FUNCIONES FRANCO
# ==============================================================

def generar_contextos(corpus, C):
  pares = []
  indices = []
  indice = 0

  while len(indices) != C:
    if indice != C/2:
      indices.append(indice)
    indice += 1 #[0,1,3,4]

  for i in range(C//2, len(corpus) - C//2):
    contexto = []
    palabra_central = corpus[i]

    for j in range(len(indices)):
      contexto.append(corpus[indices[j]])
      indices[j] += 1
    pares.append([contexto, palabra_central])

  return pares

def obtener_clave(diccionario, valor):
  for c, v in diccionario.items():
    if diccionario[c] == valor:
      return c
  
def calcular_exitacion_e_o(c_po, W, C):
  suma_Vps = 0
  for palabra in c_po[0]:
    suma_Vps += W[int(palabra)]
  return (1/C) * suma_Vps

def aplicar_softmax(u):
  u_max = np.max(u)  # estabiliza restando el máximo
  exp_u = np.exp(u - u_max)
  return exp_u / np.sum(exp_u)

def generar_one_hot(c_po, V_cardinal):
  one_hot = np.zeros(V_cardinal)
  one_hot[c_po[1]] = 1
  return one_hot

def actualizar_W(W, c_po, EH, tasa_aprendizaje, C):
  for palabra in c_po[0]:
    W[int(palabra)] = W[int(palabra)] - (tasa_aprendizaje * 1/C * EH) #1XN - 1xN
  return W

def crear_tokens(diccionario):
  tokens = {}
  with open(diccionario, 'r', encoding = 'utf-8') as dicc:
    palabras = dicc.read()
    palabras = palabras.strip('[]')
    palabras = palabras.split(',')
    for i in range(len(palabras)):
      if palabras[i] == "'":
        tokens[','] = i
      else:
        tokens[palabras[i].strip().strip("'")] = i
  return tokens

def convertir_corpus(V, corpus):
  corpus_indices = []
  with open(corpus, 'r', encoding = 'utf-8') as corpus:
    palabras = corpus.read()
    palabras = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[.,]', palabras)
    for palabra in palabras:
      if palabra in V:
        corpus_indices.append(V[palabra])
  return corpus_indices

def generar_contextos_skipgram(corpus, C):
  Pos_Cos = generar_contextos(corpus, C)
  contextos_skipgram = []
  for Po_Co in Pos_Cos:
    for palabra_Co in Po_Co[0]:
      contextos_skipgram.append([Po_Co[1], palabra_Co])
  return contextos_skipgram