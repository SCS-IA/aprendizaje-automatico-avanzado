import pdfplumber
import re

words = []

with pdfplumber.open("corpus/julio_cortazar_historias_de_cronopios_y_de_famas.pdf") as pdf:
    for page in pdf.pages[3:-1]:
        text = page.extract_text()
        if text:
            lines = text.split('\n')
            if lines[-1].strip().isdigit():
                lines = lines[:-1]
            for line in lines:
                tokens = re.findall(r"\w+|[.,!?;:]", line)
                tokens = [token.lower() for token in tokens]
                if line.endswith("."):
                    tokens[-1]= ". "
                words.extend(tokens)

print('Historias de cronopios y de famas, corpus parcial de', len(words))
print('Historias de cronopios y de famas, vocabulario parcial de', len(set(words)))

with pdfplumber.open("corpus/julio_cortazar_lucas.pdf") as pdf:
    for page in pdf.pages[5:]:
        text = page.extract_text()
        if text:
                lines = text.split('\n')
                if lines[-1].strip().isdigit():
                    lines = lines[:-1]
                lines = lines[1:]
                for line in lines:
                    tokens = re.findall(r"\w+|[.,!?;:]", line)
                    tokens = [token.lower() for token in tokens]
                    if line.endswith("."):
                        tokens[-1]= ". "
                    words.extend(tokens)

print('Se crea un vocabulario de', len(set(words)))
print('Se crea un corpus de', len(words))

# Guardar el corpus en un archivo
with open("corpus/mini_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))