import numpy as np
from aux_red_multi import generar_ventana, corpus_modificado, palabras_a_indice, indices_a_embeddings, W1
from tensorflow import keras
from tensorflow.keras import layers


import tensorflow as tf, sys



print("Py:", sys.executable, "TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No hay GPU, usando CPU")




x_gen,y1,y2_one_hot =generar_ventana(corpus_modificado, palabras_a_indice, 10, indices_a_embeddings)


W1_min, W1_max = W1.min(), W1.max()

x_escalado = 2 * ((x_gen - W1_min) / (W1_max - W1_min)) - 1
y_escalado = 2 * ((y1 - W1_min) / (W1_max - W1_min)) - 1


entradas = x_gen.shape[1]
salidas = W1.shape[1]

inputs = keras.Input(shape=(entradas,))

x = layers.Dense(1024, activation='gelu')(inputs)

x = layers.Dense(512, activation='gelu')(x)

x = layers.Dense(512, activation='gelu')(x)

x = layers.Dense(300, activation='gelu')(x)


outputs = layers.Dense(salidas, activation='tanh')(x)

multicapa_one_hot = keras.Model(inputs, outputs)

# X: embeddings concatenados o features
# Y: índice de palabra central


from tensorflow.keras.callbacks import EarlyStopping


multicapa_one_hot.compile(loss='mae', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])

multicapa_one_hot.summary()



early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

epocas = [30,30,30,30,30,30,30,30,30,30,30]

split = int(0.8 * len(x_gen))
X_train_sub = x_gen[:split]
y_train_sub = y2_one_hot[:split]
X_val = x_gen[split:]
y_val = y2_one_hot[split:]


epocas_acumuladas = 0

for i,epoca in enumerate(epocas):
    multicapa_one_hot.fit(x_escalado, y_escalado, 
                    epochs=epoca,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose = 1)
    
    epocas_acumuladas += epoca

    results = multicapa_one_hot.evaluate(x_gen, y2_one_hot, verbose=0)
    results2 = multicapa_one_hot.evaluate(X_val, y_val, verbose=0)
    print("Loss:", results[0], 'Loss Validation_data:', results2[0])
    print("Accuracy (top-1):", results[1], 'Accuracy (top-1):', results2[1])

    # Guardar solo los pesos (opcional)
    np.savez(f'multicapa_emb_epoca{epocas_acumuladas}.npz', *multicapa_one_hot.get_weights())
    
    # Guardar modelo completo
    multicapa_one_hot.save(f'multicapa_emb_model_epoca{epocas_acumuladas}.keras')

    if results2[1] > 0.9:
        break


'''with np.load('multicapa_onehot.npz') as data:
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
multicapa_one_hot.set_weights(weights)'''