import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class data_loader:
    def __init__(self, filename):
        self.filename = filename
        self.cwd = os.getcwd()
        if not os.path.exists(self.cwd + "/output"):
            os.makedirs(self.cwd + "/output")
        
        self.path = self.cwd + "/data/" + filename
        self.csv = pd.read_csv(self.path)

        self.long = []
        self.lati = []

        for i in range(len(self.csv)):
            self.long.append(self.csv['lati'][i])
            self.lati.append(self.csv['long'][i])

    def load(self):
        self.data = self.csv.values
        self.coord = np.array(self.csv[['long','lati']])
        for i in range(len(self.csv)):
            self.coord[i] = np.array([self.csv['long'][i], self.csv['lati'][i]])
        return self.data, self.coord

    def show_scatter(self):
        plt.figure(figsize=(10,8),dpi=100)
        plt.title("Scatter plot of longitude and latitude")
        plt.scatter(self.lati, self.long, s=0.3)
        plt.savefig(self.cwd+'/output/Scatter.jpg')