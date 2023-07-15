from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import sys
import soundcard as sc
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

FPS = 30
RATE = 44100

class Plot2D():
    def __init__(self):
        self.traces = dict()
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Simple Oscilloscope")
        self.win.resize(640,640)
        pg.setConfigOptions(antialias=True)
        self.canvas = self.win.addPlot()
        self.canvas.setXRange(-0.5, 0.5, padding=0)
        self.canvas.setYRange(-0.5, 0.5, padding=0)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def trace(self,name,dataset_x,dataset_y):
        if name in self.traces:
            self.traces[name].setData(dataset_x, dataset_y)
        else:
            self.traces[name] = self.canvas.plot(symbol='o', pen=None)
            self.traces[name].setSymbolSize(2)
            self.traces[name].setSymbolPen(QColor(0, 255, 0))

def update():
    global p, i
    data = mic.record(numframes=RATE//FPS)
    p.trace("sin",data[:, 0], data[:, 1])


p = Plot2D()
mics = sc.all_microphones(include_loopback=True)
for m in mics:
    if m.isloopback:
        with m.recorder(samplerate=RATE) as mic: 
            timer = QtCore.QTimer()
            timer.timeout.connect(update)
            timer.start(1000//FPS)
            p.start()