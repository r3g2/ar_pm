import sys
import matplotlib

matplotlib.use("Agg")

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QSlider
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from math import cos, sin

import matplotlib.pyplot as plt 
import numpy as np
from hmm import *



class Window(QDialog):
    def __init__(self,z_data,parent=None):
        super(Window, self).__init__(parent)

        #wid = QtWidgets.QWidget(self)
        

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        t_list = list(range(400))
        #A slider to make time variations
        self.horizontalSlider = QSlider(self)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.valueChanged.connect(self.plot)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(t_list.__len__()-1)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        #layout.addWidget(self.horizontalSlider)
        self.setLayout(layout)
        # Initializing the data
        self.x_data,self.y_data = np.meshgrid(np.arange(z_data[0].shape[1]),np.arange(z_data[0].shape[0]))
        self.x_data = self.x_data.flatten()
        self.y_data = self.y_data.flatten()
        self.data = z_data
        # Generate the chart for t=0 when the window is openned
        self.plot()

    def plot(self):
        # Read the slider value -> t = t_list[t_index]
        t_index = self.horizontalSlider.value()

        # Get the z-data for the given time index
        z_data = self.data[t_index].flatten()
        self.figure.clear()
        # Discards the old chart and display the new one
        ax = self.figure.add_subplot(111,projection='3d')
        #ax.hold(False)
        ax.bar3d(self.x_data, self.y_data, np.zeros(len(z_data)), 1, 1, z_data)

        # refresh canvas
        self.canvas.draw()

    def set_data(self,x_grid,y_grid,zdata):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.data = data

if __name__ == '__main__':


    state = np.zeros((20,20))
    state[5,5] = 1
    hmm = HMM.basic_initialization(state)
    beliefs = hmm.run(100)



    app = QApplication(sys.argv)

    main = Window(beliefs)
    print("SHOWING WINDOW")
    main.show()
    print("Almost done")
    sys.exit(app.exec_())