import re, collections
from collections import Counter
import os

# === PREPARACIÓN DEL ENTORNO ===
# Crear directorio para el corpus si no existe
os.makedirs("corpus", exist_ok=True)

# --- INSTRUCCIÓN ---
# Para usar tu propio corpus:
# 1. Crea una carpeta llamada "corpus" en el mismo directorio que este script.
# 2. Guarda tu archivo de texto dentro de esa carpeta con el nombre "corpus.txt".
# El script leerá automáticamente ese archivo.
os.makedirs("BPE", exist_ok=True)


# === 1. Leer y preparar el corpus ===
print("1. Leyendo y preparando el corpus desde 'corpus/corpus.txt'...")

corpus_path = "corpus/corpus.txt"
if not os.path.exists(corpus_path):
    print(f"\nERROR: No se encontró el archivo '{corpus_path}'.")
    print("Por favor, asegúrate de que tu corpus esté en la ubicación correcta.")
    exit()

with open(corpus_path, "r", encoding="utf-8") as f:
    palabras = [line.strip() for line in f if line.strip()]

frecuencias = Counter(palabras)

# CORRECCIÓN 1: Usar tuplas para el vocabulario.
# El vocabulario ahora usa una tupla de símbolos (caracteres) como clave.
# Esto es más robusto y evita problemas si las "palabras" contienen espacios u otros caracteres especiales.
# Ejemplo: "hola mundo" -> ('h','o','l','a',' ','m','u','n','d','o','</w>')
vocab = {tuple(list(p) + ["</w>"]): f for p, f in frecuencias.items()}


# === 2. Funciones del algoritmo BPE ===
def get_stats(vocabulario):
    """
    Calcula la frecuencia de cada par de símbolos adyacentes.
    Ahora trabaja con tuplas de símbolos como claves.
    """
    pairs = collections.defaultdict(int)
    for symbols, freq in vocabulario.items():  # 'symbols' es una tupla
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, v_in):
    """
    Fusiona un par de símbolos en todo el vocabulario.
    Ahora trabaja con tuplas de símbolos como claves.
    """
    v_out = {}
    a, b = pair
    merged_symbol = a + b
    
    for symbols, freq in v_in.items():  # 'symbols' es una tupla
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        
        v_out[tuple(new_symbols)] = freq
        
    return v_out


# === 3. Entrenamiento del BPE ===
print("\n2. Entrenando el modelo BPE...")
num_merges = 500
merges = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    
    best = max(pairs, key=pairs.get)
    merges.append(best)
    vocab = merge_vocab(best, vocab)
    
    print(f"  Fusión {i+1}/{num_merges}: {best}")

print("\nEntrenamiento completado.")


# === 4. Aplicar las fusiones al corpus original ===
print("\n3. Aplicando BPE al corpus original...")
def aplicar_bpe(palabra, merges):
    """
    Tokeniza una palabra aplicando las reglas de fusión aprendidas.
    """
    tokens = list(palabra) + ["</w>"]
    
    for a, b in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i + 1] == b:
                tokens[i:i + 2] = [a + b]
            else:
                i += 1
    # CORRECCIÓN 2: Devolvemos todos los tokens.
    # El </w> es parte de la tokenización final; si lo quitamos de forma incorrecta,
    # podemos terminar con una lista vacía, que era la causa del error.
    return tokens


# Tokenizar todo el corpus
corpus_bpe_tokens = []
for palabra in palabras:
    corpus_bpe_tokens.extend(aplicar_bpe(palabra, merges))

# === 5. Guardar resultados ===
with open("BPE/corpus_bpe.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus_bpe_tokens))

tokens_unicos = sorted(list(set(corpus_bpe_tokens)))
with open("BPE/vocab_bpe.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tokens_unicos))

with open("BPE/merges.txt", "w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a} {b}\n")

print("\n✅ Archivos generados en la carpeta 'BPE':")
print(f" - corpus_bpe.txt (corpus tokenizado), tamaño: {len(corpus_bpe_tokens)} tokens")
print(f" - vocab_bpe.txt  (vocabulario único), tamaño: {len(tokens_unicos)} tokens")
print(f" - merges.txt     (fusiones aprendidas), total: {len(merges)} fusiones")

