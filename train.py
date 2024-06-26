import numpy as np
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime

from MCNeuralNetwork import MCNeuralNetwork

n = MCNeuralNetwork()


def preprocess(f):
    image = Image.open(f)
    image = image.resize((64, 64))
    a = np.array(image) / 255.0
    a = a.reshape(64 * 64 * 3)
    return a

train = "mc-train-img/"

n = MCNeuralNetwork()

df = pd.read_csv("mc-fakes-train.csv")
df.head()
rows = 10
print("Start:",datetime.now())
for row in range(rows):
    data = df.iloc[row]
    imgname = data[1]
    fake = data[2]
    directory = train
    counter = 0
    f = os.path.join(directory, imgname)
    img = preprocess(f)
    n.train(img,fake)
    if (row % 1000 == 0):
        print(100*row/rows,"%")
print("Epoch End:", datetime.now())
        
torch.save(n.state_dict(), 'MCNN.pth')