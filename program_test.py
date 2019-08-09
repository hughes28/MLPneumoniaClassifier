from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
from sklearn.preprocessing import StandardScaler
from PIL import Image
import glob
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

image_list = []
labels = []

path = "C:/Users/Lae/PycharmProjects/MLPClassifierPokemon/Pictures/chest_xray_set/"

for root, dirs, files in os.walk(path, topdown=False):
    for files in dirs:
        print(os.path.basename(path))
        #labels.append(os.path.join(name))

print('Labels: ' + str(labels))

y = []
X = []
width = 250
height = 250


for filename in glob.glob("C:/Users/Lae/PycharmProjects/MLPClassifierPokemon/Pictures/chest_xray_set/**/*.jpeg"):
    temp_target = ''
    for label in labels:
        if label in filename:
            temp_target = label
            y.append(temp_target)
            break

    im = Image.open(filename).convert('L')
    im2 = im.resize((width, height), Image.ANTIALIAS)
    im = np.asarray(im2)
    im = im.flatten()

    X.append(im)

print(y)

print('{} pictures imported with scaling of {} x {}'.format(len(X), width, height))

y_np = np.array(y, dtype=object)
X_np = np.array(X)

accuracies = []
runs = 10

for i in range(1, runs):
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.15, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    PneumoniaClassifier = MLPClassifier(hidden_layer_sizes=7, solver='sgd', max_iter=10)

    PneumoniaClassifier.fit(X_train, y_train)
    y_pred = PneumoniaClassifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Run {} Model Accuracy: {}'.format(i, str(accuracy)))
    accuracies.append(accuracy)

accuracies = np.array(accuracies)
average_acc = np.average(accuracies) * 100
std_acc = statistics.stdev(accuracies) * 100

print('Cross-validated accuracy: {}% +/- {}%'.format(round(average_acc, 3), round(std_acc, 3)))





