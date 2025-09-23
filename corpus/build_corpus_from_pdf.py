import pdfplumber
import re

words = []


with pdfplumber.open("C:\\Users\\User\\Documents\\Aprendizaje Automatico Avanzado\\Crear Corpus\\Julio-Cortazar-Rayuela.pdf") as pdf:
    for page in pdf.pages[7:]:
        text = page.extract_text()
        if text:
            lines = text.split('\n')
            if lines[-1].strip().isdigit():
                lines = lines[:-1]
            if lines[0].strip().isdigit():
                lines = lines[1:]
            if lines[0].strip().isdigit():
                lines = lines[1:]
            if lines[-1].strip().isdigit():
                lines = lines[:-1]
            for line in lines:
                tokens = re.findall(r"\w+|[.,!?;:]", line)
                tokens = [token.lower() for token in tokens]
                if line.endswith("."):
                    tokens[-1]= ". "
                words.extend(tokens)

print('Rayuela, corpus parcial de', len(words))
print('Rayuela, vocabulario parcial de', len(set(words)))


with pdfplumber.open("C:\\Users\\User\\Documents\\Aprendizaje Automatico Avanzado\\Crear Corpus\\Julio Cortazar Todos los fuegos.pdf") as pdf:
    for page in pdf.pages[:-1]:
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
print('Todos los fuegos, corpus parcial de', len(words))
print('Todos los fuegos, vocabulario parcial de', len(set(words)))


with pdfplumber.open("C:\\Users\\User\\Documents\\Aprendizaje Automatico Avanzado\\Crear Corpus\\Historias-de-Cronopios-y-de-Famas - Julio Cortazar.pdf") as pdf:
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

with pdfplumber.open("C:\\Users\\User\\Documents\\Aprendizaje Automatico Avanzado\\Crear Corpus\\Lucas_Julio_Cortazar.pdf") as pdf:
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
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

### me anoto lo que hace pdfplumber, por si quiero usarlo en el futuro
# pdfplumber es una biblioteca de Python que permite extraer texto, tablas e imágenes de archivos PDF.
# Proporciona una interfaz sencilla para trabajar con documentos PDF y facilita la manipulación y análisis de su contenido. 
# Algunas de las características principales de pdfplumber incluyen:
# Extracción de texto: pdfplumber puede extraer texto de páginas PDF, manteniendo el formato y la estructura del documento original.
# Extracción de tablas: La biblioteca puede identificar y extraer tablas de documentos PDF, lo que facilita el análisis de datos tabulares.
# Extracción de imágenes: pdfplumber puede extraer imágenes incrustadas en archivos PDF.
# Soporte para diferentes tipos de PDF: pdfplumber es compatible con una amplia variedad de archivos PDF, incluidos aquellos con texto incrustado y aquellos que contienen solo imágenes escaneadas.
# Manipulación de páginas: La biblioteca permite acceder a páginas individuales, rotarlas, recortarlas y realizar otras operaciones de manipulación.
# Análisis de metadatos: pdfplumber puede extraer metadatos de archivos PDF, como el título, el autor y la fecha de creación.
# En resumen, pdfplumber es una herramienta útil para cualquier persona que necesite trabajar con archivos PDF en Python, ya sea para extraer información, analizar datos o manipular documentos.
# Para instalar pdfplumber, puedes usar pip:   pip install pdfplumber
# Documentación oficial: https://github.com/jsvine/pdfplumber