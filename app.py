from tkinter import filedialog
from tkinter import *
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import tkinter as Tk
from PIL import ImageTk, Image
import PIL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Model:
    def __init__(self):
        self.xpoint = 200
        self.ypoint = 200
        self.res = None
        self.filename = ''
        self.y_pred = StringVar()
        self.img = ImageTk.PhotoImage(Image.open(r"C:\Users\Lae\PycharmProjects\MLPneumoniaClassifier\Pictures\validation\validation\person21_virus_52.jpeg").resize((400, 400), PIL.Image.ANTIALIAS))
        self.textresult = Tk.Text(state='disabled', height=2, width=30)
        self.textresult.pack(side="bottom", fill=Tk.BOTH)
        self.textresult.insert(Tk.END, '')

    def calculate(self):
        x, y = np.meshgrid(np.linspace(-5, 5, self.xpoint), np.linspace(-5, 5, self.ypoint))
        z = np.cos(x ** 2 * y ** 3)
        self.res = {"x": x, "y": y, "z": z}

    def fu(self, event):
        try:
            self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
            filetypes=(("jpeg files", "*.jpeg")))
        except FileNotFoundError:
            pass
        return self.filename

    def predict(self, event):
        try:
            self.filename = filedialog.askopenfilename(initialdir=r"C:\Users\Lae\PycharmProjects\MLPneumoniaClassifier\Pictures\validation\validation", title="Select file",
            filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
            print("File uploaded")
        except FileNotFoundError:
            pass

        X_train = pickle.load(open("X_train_matrix.pkl", "rb"))
        y_train = pickle.load(open("y_train_matrix.pkl", "rb"))
        X = []

        im = Image.open(self.filename).convert('L')
        img = ImageTk.PhotoImage(im.resize((400, 400), Image.ANTIALIAS))
        im2 = im.resize((250, 250), Image.ANTIALIAS)
        im = np.asarray(im2)
        im = im.flatten()
        X_train.append(im)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = list(X_train)
        X = X_train.pop(-1)

        PneumoniaClassifier = MLPClassifier(hidden_layer_sizes=7, solver='sgd', max_iter=10)
        PneumoniaClassifier.fit(X_train, y_train)
        y_pred = PneumoniaClassifier.predict(X.reshape(1, -1))
        self.textresult.configure(state='normal')
        self.textresult.delete(1.0, Tk.END)
        self.textresult.insert(Tk.END, y_pred[0])
        self.textresult.configure(state='disabled')

    def set_y_pred(self, event):
        self.y_pred.set('hi')

class SidePanel():
    def __init__(self, root):
        self.model = Model()
        self.frame2 = Tk.Frame(root)
        self.frame2.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)
        self.predictButton = Tk.Button(self.frame2, text="Predict")
        self.predictButton.pack(side="top", fill=Tk.BOTH)
        self.predictButton.config(width=20, height=1)
        self.loadButton = Tk.Button(self.frame2, text="Load")
        self.loadButton.config(width=20, height=1)
        self.loadButton.pack(side="top", fill=Tk.BOTH)


class View:
    def __init__(self, root, model):
        self.frame = Tk.Frame(root)
        self.model = Model()
        self.filename = self.model.filename
        self.img = self.model.img
        self.frame.pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)
        self.sidepanel = SidePanel(root)

        self.sidepanel.predictButton.bind("<Button>", self.model.predict)
        self.sidepanel.loadButton.bind("<Button>", self.model.set_y_pred)

        self.canvas = Canvas(width=500, height=500, bd=0, highlightthickness=0, relief='ridge')
        self.canvas.pack()
        self.canvas.create_image(225,225, image=self.img)


class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.root.resizable(False, False)
        self.model = Model()
        self.view = View(self.root, self.model)

    def run(self):
        self.root.title("Pneumonia Detection")
        self.root.deiconify()
        self.root.mainloop()

if __name__ == '__main__':
    c = Controller()
    c.run()

