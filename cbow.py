import numpy as np
import cupy as cp
from funciones_auxiliares import cargar_corpus, inicializar_pesos, softmax_cp, generar_tuplas_central_contexto, guardar_modelo

def entrenar_cbow(archivo_corpus, nombre_pc, epocas=1, η=0.001, N=300, C=4, W1=None, W2=None, intervalo_guardado=50):
    
    corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus(archivo_corpus)
    W1, W2 = inicializar_pesos(vocab_size, N, W1, W2, cparray=True)
    indice_tuplas = generar_tuplas_central_contexto(corpus, word_to_idx, C)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        for i, (i_central, i_contextos) in enumerate(indice_tuplas):

            # ---Propagación---
            h = cp.mean(W1[i_contextos], axis=0).reshape(-1, 1)
            u = W2.T @ h
            y = softmax_cp(u)

            # ---Retropropagación---
            e = y.copy()
            e[i_central] -= 1
            W2 -= η * (h @ e.T)

            EH = W2 @ e
            W1[i_contextos] -= η * (1/C*2) * EH.T

            if i % 1000 == 0:
                print(f"Época {epoca}, Par: {i}/{total_pares}")

        print(f"Fin de época: {epoca}")

        # ---Guardado de Pesos---
        if epoca % intervalo_guardado == 0 or epoca == epocas - 1:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            guardar_modelo(nombre_archivo, W1, W2, eta=η, N=N, C=C, cparray=True)

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2 = entrenar_cbow("corpus.txt", "pcshavak-b", epocas=1000, η=0.001, N=120, C=5, intervalo_guardado=50)
W1, W2 = entrenar_cbow("corpus.txt", "pcshavak-b", epocas=1000, η=0.001, N=250, C=5, intervalo_guardado=50)