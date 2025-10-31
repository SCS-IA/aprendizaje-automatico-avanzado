import numpy as np
import cupy as cp
from funciones_auxiliares import cargar_corpus, inicializar_pesos, softmax_cp, generar_tuplas_central_contexto, guardar_modelo, cargar_modelo

def entrenar_cbow(archivo_corpus, carpeta_corpus, nombre_pc, epocas=1, η=0.001, N=300, C=4, W1=None, W2=None, intervalo_guardado=50):
    
    corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus(archivo_corpus, carpeta_corpus)
    W1, W2 = inicializar_pesos(vocab_size, N, W1, W2, cparray=True)
    W2 = W2.T
    print(f"Shape de W1 cargado: {W1.shape}")
    print(f"Shape de W2 cargado: {W2.shape}")
    print(f"Valor de N cargado: {N}")
    print(f"Dimensión N derivada de W1: {W1.shape[1]}")
    print(f"Dimensión N derivada de W2: {W2.shape[0]}")
    indice_tuplas = generar_tuplas_central_contexto(corpus, word_to_idx, C)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        E_estrella = 0
        for i, (i_central, i_contextos) in enumerate(indice_tuplas):

            # ---Propagación---
            h = cp.mean(W1[i_contextos], axis=0).reshape(-1, 1)
            u = W2.T @ h
            y = softmax_cp(u)

            # ---Error---
            E_estrella += (1 - y[i_central])

            # ---Retropropagación---
            e = y.copy()
            e[i_central] -= 1
            EH = W2 @ e

            W2 -= η * (h @ e.T)
            W1[i_contextos] -= η * (1/C*2) * EH.T

            if i % 1000 == 0:
                print(f"Época {epoca}, Par: {i}/{total_pares}", end='\r')

        print(f"Fin de época: {epoca}, con E* promedio={E_estrella/total_pares}")

        # ---Guardado de Pesos---
        if epoca % intervalo_guardado == 0 or epoca == epocas - 1:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            guardar_modelo(nombre_archivo, W1, W2, eta=η, N=N, C=C, cparray=True)

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2, N, C, eta = cargar_modelo("pesos_cbow_pcshavak-BPE10000_epoca500.npz", "weights") # Reentrenamos
# η=0.01, N=100, C=10

W1, W2 = entrenar_cbow("corpus_bpe10000.txt", "BPE10000","pcshavak-BPE10000-2ndtrain", epocas=50000, η=eta, N=N, C=C, intervalo_guardado=50, W1=W1, W2=W2)