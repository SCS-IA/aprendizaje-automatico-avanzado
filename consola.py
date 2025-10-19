import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import os
import re
from matplotlib import style

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from aux_red_multi import predecir_cbow_onehot, generar_ventana, palabras_a_indice, indices_a_palabras, indices_a_embeddings
from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\User\Documents\GitHub\Aprendizaje_Automatico\multicapa_onehot_model_epoca60.keras"
model = load_model(MODEL_PATH)


def generar_texto(texto, cantidad, model, indices_a_palabras, indices_a_embeddings, palabras_a_indice=None, topk=5):
    texto
    for i in range(cantidad):
        palabra = predecir_cbow_onehot(texto, model, indices_a_palabras, indices_a_embeddings, palabras_a_indice, topk=topk)
        if palabra is None:
            break
        if palabra in [',', '.', '?', ', y', ':', ';']:
            texto += '' + palabra
        elif palabra == ' .':
            texto += '.' + '\n\n'
        else:
            texto += ' ' + palabra
    texto = re.sub(r"\s\.\s*", ".\n\n", texto)
    return texto


def launch_cbow_gui():
    if model is None:
        print("No se pudo cargar el modelo.")
        return

    root = tk.Tk()
    root.title('Consola Julio Cortázar GPT - CBOW Red Multicapa')
    root.geometry('900x520')

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TFrame", background="#007a33")
    style.configure("TLabel", background="#007a33", foreground="#ffffff", font=("Helvetica", 11))
    style.configure("TButton",
                    background="#ffffff",
                    foreground="#007a33",
                    font=("Helvetica", 10, "bold"))
    style.map("TButton",
            background=[("active", "#5cb85c")],
            foreground=[("active", "#ffffff")])
    style.configure("TScale", background="#007a33", troughcolor="#5cb85c")

    mainframe = ttk.Frame(root, padding='8')
    mainframe.pack(fill='both', expand=True)

    left = ttk.Frame(mainframe)
    left.grid(column=0, row=0, sticky='nsew')

    ttk.Label(left, text='Frase inicial:').grid(column=0, row=0, sticky='w')
    seed_entry = ttk.Entry(left, width=80)
    seed_entry.grid(column=0, row=1, columnspan=3, sticky='we', pady=4)

    ttk.Label(left, text='Cantidad de palabras a generar:').grid(column=0, row=2, sticky='w')
    cantidad_spin = ttk.Spinbox(left, from_=1, to=200, width=6)
    cantidad_spin.set(20)
    cantidad_spin.grid(column=1, row=2, sticky='w', padx=(8,0))

    ttk.Label(left, text='Top-k aleatorio:').grid(column=0, row=3, sticky='w')
    top_k_spin = ttk.Spinbox(left, from_=1, to=200, width=6)
    top_k_spin.set(5)
    top_k_spin.grid(column=1, row=3, sticky='w', padx=(8,0))

    output_box = scrolledtext.ScrolledText(left, wrap='word', height=20, width=80)
    output_box.grid(column=0, row=4, columnspan=3, sticky='nsew', pady=8)

    def on_generate():
        seed = seed_entry.get().strip()
        if not seed:
            output_box.delete('1.0', tk.END)
            output_box.insert(tk.END, 'Introduce una frase inicial.')
            return
        try:
            cantidad = int(cantidad_spin.get())
        except Exception:
            cantidad = 20

        resultado = generar_texto(seed, cantidad, model, indices_a_palabras, indices_a_embeddings, palabras_a_indice, topk=int(top_k_spin.get()))

        output_box.delete('1.0', tk.END)
        output_box.insert(tk.END, resultado)

    def on_clear():
        seed_entry.delete(0, tk.END)
        output_box.delete('1.0', tk.END)

    generate_btn = ttk.Button(left, text='Generar', command=on_generate)
    generate_btn.grid(column=0, row=5, sticky='w')

    clear_btn = ttk.Button(left, text='Limpiar', command=on_clear)
    clear_btn.grid(column=1, row=5, sticky='w')

    right = ttk.Frame(mainframe)
    right.grid(column=1, row=0, sticky='ns', padx=(12,0))

    img_label = ttk.Label(right, text='Julio Cortázar GPT')
    img_label.pack(anchor='nw')

    canvas_label = ttk.Label(right, text='(no hay imagen)')
    canvas_label.pack(pady=(6,8))

    FIXED_IMAGE = os.path.join('CBOW', 'Red_multicapa', 'images.webp')
    def load_fixed_image():
        if not os.path.exists(FIXED_IMAGE):
            canvas_label.config(text=f'No se encontró {FIXED_IMAGE}')
            return
        if not PIL_AVAILABLE:
            canvas_label.config(text='Pillow no disponible; instala pillow para mostrar imágenes')
            return
        try:
            img = Image.open(FIXED_IMAGE)
            img.thumbnail((280, 280))
            tkimg = ImageTk.PhotoImage(img)
            canvas_label.image = tkimg
            canvas_label.config(image=tkimg, text='')
        except Exception as e:
            canvas_label.config(text=f'No se pudo cargar imagen:\n{e}')

    load_fixed_image()

    authors_frame = ttk.Frame(right)
    authors_frame.pack(fill='x', pady=(8,0))

    authors_label = ttk.Label(authors_frame, text='Autores: Cisnero Matías, Seivane Nicolás, Serafini Franco', wraplength=260, justify='left')
    authors_label.pack(side='left', anchor='nw')

    logo_path = os.path.join('CBOW', 'Red_multicapa', 'logo_universidad.png')
    fallback_logo = None
    try:
        possible = os.path.join(os.path.dirname(os.__file__), '..', 'share', 'jupyter', 'kernels', 'python3', 'logo-64x64.png')
        if os.path.exists(possible):
            fallback_logo = possible
    except Exception:
        fallback_logo = None

    final_logo = logo_path if os.path.exists(logo_path) else (fallback_logo if fallback_logo and os.path.exists(fallback_logo) else None)
    if final_logo and PIL_AVAILABLE:
        try:
            logo_img = Image.open(final_logo)
            logo_img.thumbnail((100, 100))
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_container = ttk.Frame(right)
            logo_container.pack(fill='both', expand=True, pady=(8,0))

            max_width = 300
            w, h = logo_img.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                logo_img = logo_img.resize((max_width, new_h), Image.ANTIALIAS)
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(logo_container, image=logo_tk)
            logo_label.image = logo_tk
            logo_label.pack(side='bottom', anchor='se')
        except Exception:
            pass
    else:
        no_logo_label = ttk.Label(logo_container, text='(logo no disponible)')
        no_logo_label.pack(side='bottom', anchor='se')

    mainframe.columnconfigure(0, weight=1)
    mainframe.columnconfigure(1, weight=0)
    mainframe.rowconfigure(0, weight=1)
    left.columnconfigure(0, weight=1)
    left.rowconfigure(4, weight=1)

    root.mainloop()


if __name__ == '__main__':
    launch_cbow_gui()