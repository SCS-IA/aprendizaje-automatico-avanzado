# 📚 NLP Models

Proyecto realizado por [Matías Cisnero](https://github.com/matias-cisnero), [Nicolas Seivane](https://github.com/NicolasSeivane) y [Franco Serafini](https://github.com/francoSerafini).  

Este proyecto implementa **modelos de representación de palabras desde cero**, utilizando únicamente **NumPy** y **CuPy (CUDA)** para cálculo matricial acelerado en GPU. Es un ejercicio práctico de **aprendizaje automático avanzado**, diseñado para explorar implementaciones desde cero y comprender el funcionamiento interno de **Word2Vec**.  

* 📓 Implementación pura sin frameworks de deep learning
* ⚡ Uso de **CuPy** (similar a NumPy) para acelerar el entrenamiento en GPU
* 🧠 Modelos implementados: `CBOW` y `Skip-gram`

## 🔍 Datos utilizados

* **Corpus total**: 310,347 palabras
* **Vocabulario**: 30,283 palabras únicas
* **Fuente de datos**: libros de **Julio Cortázar**

  * *Historias de cronopios y de famas*
  * *Todos los fuegos el fuego*
  * *Rayuela*
  * *Un tal Lucas*

## ⚙️ Tecnologías utilizadas

- <a href="https://www.python.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/640px-Python_logo_and_wordmark.svg.png" alt="Logo Python" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Lenguaje de programación utilizado para todo el proyecto</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://numpy.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/640px-NumPy_logo_2020.svg.png" alt="Logo NumPy" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Biblioteca para operaciones numéricas en Python</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://cupy.dev/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/CuPy_Logo.png/640px-CuPy_Logo.png" alt="Logo CuPy" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– CuPy para cómputo en GPU con</span> 
  <a href="https://www.nvidia.com/" target="_blank">
    <img src="https://cdn.worldvectorlogo.com/logos/nvidia-7.svg" alt="Logo CUDA" width="20" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;"> [CUDA]</span>
  <div style="margin-bottom: 10px;"></div>

## 🧠 ¿Qué hace este proyecto?

* Procesa el corpus de Cortázar para construir el vocabulario.
* Implementa **desde cero** los algoritmos:

  * **CBOW (Continuous Bag of Words)**
  * **Skip-gram**
* Entrena los modelos para obtener **vectores de palabras** (embeddings) de menor dimensión.
* Permite analizar relaciones semánticas a partir de la proximidad en el espacio vectorial.

## 🚀 ¿Cómo lo uso?

1. Clonar este repositorio:

   ```bash
   git clone https://github.com/matias-cisnero/nlp-models.git
   cd nlp-models
   ```

2. Ejecutar el entrenamiento de CBOW o Skip-gram:

   ```bash
   python cbow.py
   python skipgram.py
   ```

3. Revisar los embeddings generados para explorar representaciones de palabras.

## 📁 Estructura del proyecto

```
nlp-models/
├── funciones auxiliares.py
├── cbow.py                <- Modelo CBOW (CuPy)
├── cbow_negative_sampling.py   <- Modelo CBOW con negative sampling (NumPy)
├── skipgram.py            <- Modelo Skip-gram (CuPy)
├── skipgram_negative_sampling.py   <- Modelo Skip-gram con negative sampling (NumPy)
├── probar_modelos.ipynb   <- Notebook para interactuar con los modelos
├── corpus/                <- Corpus de entrenamiento
├── weights/                <- Pesos de las matrices W y W'
├── docs/           <- Documentos y presentaciones
└── README.md
```

---

## 🧠 Conceptos clave

* **CBOW**: predice una palabra a partir de su contexto.
* **Skip-gram**: predice el contexto a partir de una palabra.
* **Embeddings**: vectores de baja dimensión que representan palabras en un espacio semántico.
* **NumPy/CuPy**: librerías para cómputo numérico, la segunda con soporte GPU.

---

