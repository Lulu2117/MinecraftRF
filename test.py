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

train = "mc-test-img/"

n = MCNeuralNetwork()
n.load_state_dict(torch.load('MCNN.pth'))

df = pd.read_csv("mc-fakes-test.csv")
df.head()
correct = 0
rows = 100
print("Start:",datetime.now())
for row in range(rows):
    data = df.iloc[row]
    imgname = data[1]
    fake = data[2]
    directory = train
    counter = 0
    f = os.path.join(directory, imgname)
    img = preprocess(f)

    output = n.forward(img).detach().numpy()
    
    guess = 0
    if output[0] > 0.5:
        guess = 1
    if guess == fake:
        correct += 1

print("Accuracy:",100*correct/rows,"%")
print("Epoch End:", datetime.now())
        
torch.save(n.state_dict(), 'MCNN.pth')