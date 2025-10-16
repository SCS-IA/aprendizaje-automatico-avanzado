import re, collections
from collections import Counter

# === 1. Leer y preparar el corpus ===
with open("corpus/corpus.txt", "r", encoding="utf-8") as f:
    palabras = [line.strip() for line in f if line.strip()]

# Contar frecuencias
frecuencias = Counter(palabras)

# Convertir al formato inicial de BPE
vocab = {" ".join(list(p)) + " </w>": f for p, f in frecuencias.items()}


# === 2. Funciones del algoritmo BPE ===
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


# === 3. Entrenamiento del BPE ===
num_merges = 500
merges = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    merges.append(best)
    vocab = merge_vocab(best, vocab)
    print(f"Merging {best}")


# === 4. Aplicar las fusiones al corpus original ===
def aplicar_bpe(palabra, merges):
    tokens = list(palabra) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i + 1] == b:
                tokens[i:i + 2] = [a + b]
            else:
                i += 1
    return tokens[:-1]  # quitar </w>


# Tokenizar el corpus y aplanarlo en líneas
corpus_bpe_tokens = []
for palabra in palabras:
    corpus_bpe_tokens.extend(aplicar_bpe(palabra, merges))

# === 5. Guardar resultados ===
# Corpus: una línea por token
with open("BPE/corpus_bpe.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus_bpe_tokens))

# Vocabulario: conjunto de tokens únicos
tokens_unicos = sorted(set(corpus_bpe_tokens))
with open("BPE/vocab_bpe.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(tokens_unicos))

# Fusiones aprendidas
with open("BPE/merges.txt", "w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a} {b}\n")

print("\n✅ Archivos generados:")
print(f" - corpus_bpe.txt  (una línea por token BPE), tamaño: {len(corpus_bpe_tokens)}")
print(f" - vocab_bpe.txt   (vocabulario único), tamaño: {len(tokens_unicos)}")
print(f" - merges.txt      (fusiones aprendidas)")
