import numpy as np
import cupy as cp
from funciones_auxiliares import cargar_corpus, inicializar_pesos, softmax_cp2, generar_tuplas_central_contexto, guardar_modelo

def entrenar_cbow(archivo_corpus, nombre_pc, epocas=1, η=0.001, N=300, C=4, W1=None, W2=None, intervalo_guardado=50):
    
    corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus(archivo_corpus)
    W1, W2 = inicializar_pesos(vocab_size, N, W1, W2, cparray=True)
    indice_tuplas = generar_tuplas_central_contexto(corpus, word_to_idx, C)
    total_pares = len(indice_tuplas)

    print(f"Comienzo de entrenamiento con {epocas} epocas.")
    for epoca in range(epocas):
        E = 0
        E_estrella = 0

        for i, (i_central, i_contextos) in enumerate(indice_tuplas):

            # ---Propagación---
            h = cp.mean(W1[i_contextos], axis=0).reshape(-1, 1)
            u = W2.T @ h
            y, sum_exp = softmax_cp2(u)

            # ---Error---
            E += (np.log(sum_exp) - u[i_central])
            E_estrella += (1 - y[i_central])

            # ---Retropropagación---
            e = y.copy()
            e[i_central] -= 1
            EH = W2 @ e

            W2 -= η * (h @ e.T)
            W1[i_contextos] -= η * (1/C*2) * EH.T

            # ---Print---
            if i % 1000 == 0:
                progreso = int((i / total_pares) * 30)
                barra = "█" * progreso + "-" * (30 - progreso)
                porc = (i / total_pares) * 100
                print(f"  [{barra}] {porc:6.2f}%  ({i}/{total_pares})", end='\r')

        print(f"Fin de época: {epoca}, con E={float(E):.4f}|E/N={float(E/total_pares):.4f} y E*={float(E_estrella):.4f}|E*/N={float(E_estrella/total_pares):.4f}")

        # ---Guardado de Pesos---
        if epoca % intervalo_guardado == 0 or epoca == epocas - 1:
            nombre_archivo = f'pesos_cbow_{nombre_pc}_epoca{epoca}.npz'
            guardar_modelo(nombre_archivo, W1, W2, eta=η, N=N, C=C, cparray=True)

    print(f"Entrenamiento con {epocas} terminado.")
    return W1, W2

W1, W2 = entrenar_cbow("mini_corpus.txt", "pcshavak-mini", epocas=100, η=0.01, N=50, C=4, intervalo_guardado=10)