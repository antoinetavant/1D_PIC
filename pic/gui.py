from tkinter import Button, Frame, Text, INSERT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numba import jit


class GUI(Frame):
    def __init__(self, master=None):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.validated = None

        self.master = master

        self.master.title(" Please read carfully")
        self.pack(fill="both", expand=1)

        self.close_button = Button(master, text="Close",
                                   command=self.client_exit)
        self.close_button.pack()

    def ok(self):
        self.validated = True

    def not_ok(self):
        self.validated = False

    def client_exit(self):
        self.quit()

    def add_text(self, str):
        text = Text(self.master, height=1)
        text.insert(INSERT, str)
        text.pack()

    def add_button(self, text, command):
        self.greet_button = Button(self.master, text=text, command=command)
        self.greet_button.pack()


class LivePlot():
    """Object that help the plot of the informations """

    def __init__(self, tab_x, tab_v, strList=["ne"]):
        """init all """

        self.tabx = tab_x
        self.tabv = tab_v
        self.strList = strList
        self.Nplots = len(self.strList)
        self.Nrows = 2
        self.Ncols = max(int((self.Nplots + 1) // 2), 1)

        # creat the figure and the axes
        self.fig, self.axarr = self.creataxes()

        # create the line
        self.Lines = [Line2D([], []) for i in range(self.Nplots)]

        # add the title and the x limites
        for ax, strvalue in zip(self.axarr, self.strList):
            ax.set_title(strvalue)
            ax.set_xlim(self.tabx[0], self.tabx[-1])

        # add the line to the axes
        for ax, line in zip(self.axarr, self.Lines):
            ax.add_line(line)

        plt.show()

    def creataxes(self):

        plt.ion()   # Note this correction
        fig = plt.figure()

        axarr = [0 for i in range(self.Nplots)]

        for i in range(self.Nplots):
            axarr[i] = fig.add_subplot(self.Nrows, self.Ncols, i+1)

        return fig, np.array(axarr)

    def updatevalue(self, data, nt, Nt, dT):

        for line, st in zip(self.Lines, self.strList):
            line.set_data(self.tabx, smooth(data[st]))

        plt.suptitle("Nt = {:1.1e} over {:1.1e}, ".format(nt, Nt) +
                     "t = {:2.2e} $\\mu s$".format(nt*dT*1e6), fontsize=12)
        plt.draw()
        plt.pause(0.00001)  # Note this correction


@jit("f8[:](f8[:])")
def smooth(x):

    y = np.zeros_like(x)
    N = len(x)
    y[[0, -1]] = x[[0, -1]]

    for i in np.arange(1, N-2):
        y[i] = (2*x[i] + x[i-1] + x[i+1])/4

    return y
