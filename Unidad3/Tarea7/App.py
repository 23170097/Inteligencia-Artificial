import os
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageDraw


def preprocesar_imagen(img):
    arr = np.array(img.convert("L"), dtype=np.float32)
    if arr.max() < 10:
        return None

    arr /= 255.0
    arr[arr < 0.15] = 0.0

    filas, cols = np.nonzero(arr)
    recorte = arr[filas.min():filas.max() + 1, cols.min():cols.max() + 1]
    h, w = recorte.shape

    escala = 20.0 / max(h, w)
    nuevo_w, nuevo_h = max(1, round(w * escala)), max(1, round(h * escala))
    img_redim = Image.fromarray((recorte * 255).astype(np.uint8)).resize(
        (nuevo_w, nuevo_h), Image.Resampling.LANCZOS
    )
    datos_20 = np.array(img_redim, dtype=np.float32) / 255.0

    lienzo = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nuevo_h) // 2, (28 - nuevo_w) // 2
    lienzo[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = datos_20

    suma = lienzo.sum()
    if suma > 0:
        yy, xx = np.indices(lienzo.shape)
        cy, cx = (yy * lienzo).sum() / suma, (xx * lienzo).sum() / suma
        lienzo = np.roll(lienzo, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))

    return lienzo.reshape(1, 784)


class AppReconocedor:
    def __init__(self, root, ruta_modelo):
        self.root = root
        self.root.title("Clasificador de Dígitos (0-9)")
        self.root.resizable(False, False)

        self.modelo = joblib.load(ruta_modelo)
        self.tam = 280
        self.grosor = 18
        self.ultimo_punto = None
        self.trabajo_pendiente = None

        self.img = Image.new("L", (self.tam, self.tam), 0)
        self.draw = ImageDraw.Draw(self.img)

        tk.Label(root, text="Dibuja un número (0-9)", font=("Arial", 16, "bold")).pack(pady=(10, 5))

        self.canvas = tk.Canvas(root, width=self.tam, height=self.tam, bg="black", cursor="cross")
        self.canvas.pack(padx=15, pady=5)

        self.lbl_pred = tk.Label(root, text="Predicción: -", font=("Arial", 18, "bold"))
        self.lbl_pred.pack(pady=(8, 2))

        self.lbl_conf = tk.Label(root, text="Confianza: -", font=("Arial", 12))
        self.lbl_conf.pack(pady=(0, 10))

        tk.Button(root, text="Limpiar", font=("Arial", 12), width=14, command=self.limpiar).pack(pady=(0, 15))

        self.canvas.bind("<B1-Motion>", self.dibujar)
        self.canvas.bind("<ButtonRelease-1>", self.soltar)

    def dibujar(self, evento):
        r = self.grosor // 2
        if self.ultimo_punto is not None:
            x0, y0 = self.ultimo_punto
            self.canvas.create_line(x0, y0, evento.x, evento.y, fill="white", width=self.grosor, capstyle=tk.ROUND)
            self.draw.line([x0, y0, evento.x, evento.y], fill=255, width=self.grosor)

        self.canvas.create_oval(evento.x - r, evento.y - r, evento.x + r, evento.y + r, fill="white", outline="white")
        self.draw.ellipse([evento.x - r, evento.y - r, evento.x + r, evento.y + r], fill=255)
        self.ultimo_punto = (evento.x, evento.y)

        if self.trabajo_pendiente is None:
            self.trabajo_pendiente = self.root.after(200, self.predecir)

    def soltar(self, evento):
        self.ultimo_punto = None
        self.predecir()

    def predecir(self):
        self.trabajo_pendiente = None
        entrada = preprocesar_imagen(self.img)
        if entrada is None:
            return

        digito = self.modelo.predict(entrada)[0]
        confianza = float(np.max(self.modelo.predict_proba(entrada))) * 100

        self.lbl_pred.config(text=f"Predicción: {digito}")
        self.lbl_conf.config(text=f"Confianza: {confianza:.1f}%")

    def limpiar(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (self.tam, self.tam), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.ultimo_punto = None
        self.lbl_pred.config(text="Predicción: -")
        self.lbl_conf.config(text="Confianza: -")


if __name__ == "__main__":
    carpeta = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(carpeta, "digit_mlp.joblib")

    if not os.path.exists(ruta):
        raise SystemExit("Error: Primero ejecuta 'EntrenarNumero.py' para generar 'digit_mlp.joblib'.")

    ventana = tk.Tk()
    AppReconocedor(ventana, ruta)
    ventana.mainloop()