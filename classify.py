from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from PIL import Image
import pickle
import numpy as np

def classify(filename):
    X_train = pickle.load(open("C:/Users/Lae/PycharmProjects/MLPClassifierPokemon/X_train_matrix.pkl","rb"))
    y_train = pickle.load(open("C:/Users/Lae/PycharmProjects/MLPClassifierPokemon/y_train_matrix.pkl","rb"))
    X = []

    im = Image.open(filename).convert('L')
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

    return y_pred

classify(r'C:\Users\Lae\PycharmProjects\MLPClassifierPokemon\Pictures\validation\validation\person21_virus_52.jpeg')


