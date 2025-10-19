import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

def cargar_modelo_completo(nombre_archivo='pesos_cbow_pc2_epoca0.npz'):
    
    try:
        data = np.load(nombre_archivo)

        W1 = data['W1']
        W2 = data['W2']

        N = data['N'].item()
        C = data['C'].item()
        eta = data['eta'].item()

        print()

        return W1, W2, N, C, eta

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None, None, None, None
    
def generar_ventana(corpus, palabras_a_indice, contexto, indices_a_embeddings):
    indices = range(contexto, len(corpus))
    indices_contexto = range(-contexto, 0)

    contexto_a_central = {}
    indice_a_palabra = {v: k for k, v in palabras_a_indice.items()}
    X = []
    Y = []
    Y2 = []
    for i in indices:
        palabra_central = palabras_a_indice[corpus[i]]
        contexto_actual = tuple(palabras_a_indice[corpus[i+j]] for j in indices_contexto)

        # Si ya existe el mismo contexto pero con otra palabra central
        if contexto_actual in contexto_a_central:
            if contexto_a_central[contexto_actual] != palabra_central:
                continue
        else:
            contexto_a_central[contexto_actual] = palabra_central
        
        ventana = np.concatenate([indices_a_embeddings[idx] for idx in contexto_actual], axis=0)
        X.append(ventana.flatten())          # Aplanamos para que quede 1D
        Y.append(indices_a_embeddings[palabra_central])
        Y2.append(palabra_central)
    return np.array(X), np.array(Y), np.array(Y2, dtype=np.int32)


with open("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\Evaluacion_Modelos\\corpus_junto2_todos_los_fuegos.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

corpus_modificado = words.copy()

palabras_a_indice = {}
indices_a_palabras = {}
diccionario_onehot = {}
diccionario_onehot_a_palabra = {}
diccionario_conteo = {}
indices_a_embeddings = {}

W1, W2,N, C, eta = cargar_modelo_completo("C:\\Users\\User\\Documents\\GitHub\\Aprendizaje_Automatico\\pesos_cbow_neg_epoca1500_contexto_5.npz")
if W1 is None:
    print('aca esta el problema')

for token in words:
    if token not in palabras_a_indice:
        index = len(palabras_a_indice)
        palabras_a_indice[token] = index
        indices_a_palabras[index] = token
        indices_a_embeddings[index] = W1[index].reshape(1,-1)
        diccionario_conteo[token] = 1 
    else:
        diccionario_conteo[token] += 1 


cardinal_V = len(palabras_a_indice)

for token, idx in list(palabras_a_indice.items()):

    one_hot_vector = np.zeros(cardinal_V)
    one_hot_vector[idx] = 1
    diccionario_onehot[token] = one_hot_vector
    diccionario_onehot_a_palabra[str(one_hot_vector)] = token

def tokenizar_por_vocab(texto, vocab, indices = False):
    palabras = texto.lower()
    palabras = re.findall(r'\w+|[^\w\s]', palabras, flags=re.UNICODE) # tokenización básica por espacios
    tokens = []
    i = 0
    n = len(palabras)

    while i < n:
        cand_final = None
        for j in range(n, i, -1):
            cand = " ".join(palabras[i:j])
            if cand in vocab:
                cand_final = cand
                i = j  
                break
        
        if not cand_final:
            cand_final = palabras[i]
            if cand_final not in vocab:
               print(f'palabra: [{cand_final}] no esta en voabulario') 
               return None
               
            i += 1

        if indices is False:
            tokens.append(cand_final)
        else:
            tokens.append(palabras_a_indice[cand_final])
    return tokens

def predecir_cbow_onehot(palabras, modelo, indice_a_palabras, indices_a_embeddings, palabras_a_indice=None, topk=5):

    # usar el vocab proporcionado o la global
    if palabras_a_indice is None:
        try:
            palabras_a_indice = globals().get('palabras_a_indice')
        except Exception:
            palabras_a_indice = None

    # tokenizar y obtener índices (tokenizar_por_vocab soporta indices=True)
    tokens_idx = tokenizar_por_vocab(palabras, palabras_a_indice, indices=True)

    if tokens_idx is None or len(tokens_idx) == 0:
        return None

    # mantener sólo las últimas 10 palabras; si hay menos, rellenar con la última palabra
    if len(tokens_idx) < 10:
        tokens_idx = tokens_idx + [tokens_idx[-1]] * (10 - len(tokens_idx))
    else:
        tokens_idx = tokens_idx[-10:]

    # construir la ventana concatenando vectores
    try:
        ventana = np.concatenate([indices_a_embeddings[idx] for idx in tokens_idx]).flatten()
    except Exception as e:
        print(f"Error al construir la ventana de entrada: {e}")
        return None

    # predecir (se asume que el modelo devuelve probabilidades)
    pred = modelo.predict(ventana.reshape(1, -1), verbose=0)
    probs = np.asarray(pred).flatten()

    # obtener top-k
    candidatos = np.argsort(-probs)
    topk_indices = candidatos[:topk]
    top1 = np.random.choice(topk_indices)
    palabra = indice_a_palabras[top1]
    return palabra

def predecir_cbow_embedding(palabras, modelo, indice_a_palabras, W, palabras_a_indice=None, topk=5):

    if palabras_a_indice is None:
        try:
            palabras_a_indice = globals().get('palabras_a_indice')
        except Exception:
            palabras_a_indice = None

    # tokenizar y obtener índices
    tokens_idx = tokenizar_por_vocab(palabras, palabras_a_indice, indices=True)

    if not tokens_idx:
        return None

    # mantener sólo las últimas 10 palabras; si hay menos, rellenar
    if len(tokens_idx) < 10:
        tokens_idx = tokens_idx + [tokens_idx[-1]] * (10 - len(tokens_idx))
    else:
        tokens_idx = tokens_idx[-10:]

    # concatenar embeddings directamente desde W
    try:
        ventana = np.concatenate([W[idx] for idx in tokens_idx]).flatten()
    except Exception as e:
        print(f"Error al construir la ventana de entrada: {e}")
        return None

    # predecir embedding
    try:
        pred_emb = modelo.predict(ventana.reshape(1, -1), verbose=0)
        pred_emb = np.asarray(pred_emb).flatten()
    except Exception as e:
        print(f"Error al predecir el embedding: {e}")
        return None

    # calcular similitudes con todos los embeddings
    sims = cosine_similarity(pred_emb.reshape(1, -1), W)[0]

    # top-k índices más similares
    topk_idx = np.argsort(-sims)[:topk]

    # elegir una palabra aleatoria entre las top-k
    top1 = np.random.choice(topk_idx)
    palabra_predicha = indice_a_palabras[top1]

    return palabra_predicha