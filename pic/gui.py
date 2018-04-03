from tkinter import Tk, Label, Button, Frame, Text, INSERT, END, BOTH

class GUI(Frame):
    def __init__(self, master = None):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)
        self.validated = None

        self.master = master

        self.master.title(" Please read carfully")
        self.pack(fill="both", expand=1)

        self.close_button = Button(master, text="Close", command=self.client_exit)
        self.close_button.pack()

    def ok(self):
        self.validated = True

    def not_ok(self):
        self.validated = False

    def client_exit(self):
        self.quit()

    def add_text(self, str):
        text = Text(self.master, height = 1)
        text.insert(INSERT, str)
        text.pack()

    def add_button(self,text,command):
        self.greet_button = Button(self.master, text=text, command=command)
        self.greet_button.pack()
