#kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
import os

class QuantumLabScreen(BoxLayout):
    PYTHON_NAME = "python3"
    def run_python(self, filename):
        global cmd
        cmd = self.PYTHON_NAME + " " + filename
        os.system(cmd)

class QuantumLabApp(App):
    def build(self):
        screen = QuantumLabScreen()
        return screen

if __name__ == "__main__":
    app = QuantumLabApp()
    app.run()
