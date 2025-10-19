import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funciones_auxiliares import embeber_datos, cargar_corpus, cargar_modelo

corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus("corpus.txt", "corpus")
W1, W2, N, C, eta = cargar_modelo("pesos_cbow_pcshavak-c_epoca1600.npz", "relevant_weights") # MEJOR HASTA EL MOMENTO
ventana = 10
x_train, y_train = embeber_datos(corpus, W1, word_to_idx, ventana)
tam_salida = len(y_train)


# Datos de entrenamiento
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# Definición del modelo
#model = keras.models.load_model("./PMC/modelo_pmc2.keras")
model = Sequential()
model.add(Dense(512, input_shape=(N*ventana,)))
model.add(Dense(256, activation='gelu'))
model.add(Dense(128, activation='gelu'))
model.add(Dense(tam_salida, activation='softmax'))

# Compilación

model.compile(optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Entrenamiento
historia = model.fit(x_train, y_train, epochs=10)

#Cargo el modelo ya entrenado
#historia = keras.models.load_model("")

print(f"Error minimo: {min(historia.history['loss'])}")

# Gráfico de la pérdida
plt.plot(historia.history['loss'], label='Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (loss)')
plt.title('Evolución del error')
plt.legend()
plt.grid()
plt.show()

# Guardar el modelo entrenado
model.save("modelo_pmcSM.keras")
print("Modelo guardado")