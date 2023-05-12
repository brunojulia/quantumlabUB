import matplotlib

from functools import partial

import kivy
from kivy.app import App
from kivy.uix.label import Label

from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.uix.checkbox import CheckBox


from kivy.properties import ObjectProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
#matplotlib.use('module://kivy.garden.matplotlib.backend_kivyagg')
# Should this be used? Can't save to file?

import numpy as np
from numba import jit
import numba
import crankNicolson.crankNicolson2D as mathPhysics
import crankNicolson.animate as animate
import matplotlib.pyplot as plt
import time

####
# https://stackoverflow.com/questions/70629758/kivy-how-to-pass-arguments-to-a-widget-class
#####

class BoolCheckBox(CheckBox):
    pass

class GlobalVariable(GridLayout):
    def __init__(self, source, names, num=0, **kwargs):
        super(GlobalVariable, self).__init__(cols=3,**kwargs)
        self.num = num
        self.names = names
        self.source = source # List of globals, numpy array

        self.label = Label(text="Var.\n{}".format(num), size_hint_x=0.2)

        self.nameText = DataInput(attribute="names", index=self.num, holder=self, condition="unique", multiline=False, size_hint_x=0.4)

        self.valText = DataInput(attribute="source", index=self.num, holder=self, multiline=False, size_hint_x=0.4)


        #self.layout = GridLayout(cols=3, size=(100, 100))
        self.add_widget(self.label)
        self.add_widget(self.nameText)
        self.add_widget(self.valText)
        #self.add_widget(self.layout)

    """def on_value(self, instance, value):
        self.source[self.num] = self.value"""

class GlobalVariablesPopup(Popup):
    nVar = 16
    def __init__(self, window, **kwargs):
        self.window = window  # Window holds information such as QuantumSystem and Animation
        super(GlobalVariablesPopup, self).__init__(**kwargs)

        self.layout = GridLayout(cols=2)
        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        #print(self.window.paramNames)
        for i in range(self.nVar):
            self.layout.add_widget(GlobalVariable(self.window.extra_param, self.window.paramNames, num=i))
        self.add_widget(self.layout)

    def on_dismiss(self):
        super(GlobalVariablesPopup, self).on_dismiss()
        # We need to wait to make sure all DataInputs finish on unfocus
        Clock.schedule_once(lambda *param: self.window.setVarDict())


class DataInput(TextInput):
    """
    Holds an attribute which it can change/modify. The attribute is passed by value,
    so to modify it in the class instance it refers to that class instance needs to be passed.
    Then it can be modified with vars(classInstance) / classInstance.__dict__.
    Which returns a dictionary of variables, mutable.
    """
    def __init__(self, attribute = None, index=None, holder = None, condition = None, **kwargs):
        super(DataInput, self).__init__(**kwargs)
        self.attribute = attribute
        self.index = index
        self.holder = holder
        self.condition = condition
        Clock.schedule_once(self.set_self)
        # .kv can't take data from another class during init. Everything needs to be init first
        # That's why the delay

    def set_self(self, dt):
        # Attribute is set in kivy language
        #print(id(self.attribute)) Check
        self.attributeVal = vars(self.holder)[self.attribute] if self.index is None else \
                            vars(self.holder)[self.attribute][self.index]
        self.type = type(self.attributeVal)
        self.text = str(self.attributeVal)
        self.copy = self.text

    def _on_focus(self, instance, value, *largs):
        # When unfocusing also apply changes
        super(DataInput, self)._on_focus(instance, value, *largs)
        if value == False and self.text != self.copy:
            self.on_text_validate()


    def on_text_validate(self):
        # Enter
        try:
            self.attributeVal = self.type(self.text)
            # Try first to see if it can be converted sucessfully
        except:
            #print("Couldn't convert properly")
            TextPopup("Invalid Input!").open()
            self.text = self.copy
            return
        if self.conditionHolds(self.attributeVal):
            if self.index is None:
                vars(self.holder)[self.attribute] = self.type(self.text)
            else:
                vars(self.holder)[self.attribute][self.index] = self.type(self.text)
            self.copy = self.text
        else:
            self.text = self.copy

    def conditionHolds(self, val):
        if self.condition == None: return True
        elif self.condition == "notNothing":
            if val == "": return False
        elif self.condition == "unique":
            ### Not too general
            ### We assume here no name is allowed to be repeated as a name
            if val != "" and val in vars(self.holder)[self.attribute]:
                TextPopup("Can't repeat!", title="Warning").open()
                return False
        elif self.condition == "nonnegative":
            if val < 0:
                TextPopup("Must be nonnegative!", title="Warning").open()
                return False
        elif self.condition == "positive":
            if val <= 0:
                TextPopup("Must be posiive!", title="Warning").open()
                return False
        elif self.condition.startswith("range"):

            left = self.condition.split('-')[1]
            right = self.condition.split('-')[2]
            if val < float(left) or float(right) < val:
                TextPopup("Must be between {0} and {1}".format(left, right), title="Warning").open()
                return False
        return True


class FunctionInput(TextInput):
    """Expected to hold string python expression which can be converted into a function."""

    def __init__(self, functionName=None, definitionName=None, varDict={}, holder=None, condition=None, jit = False, **kwargs):
        super(FunctionInput, self).__init__(**kwargs)
        self.functionName = functionName
        self.definitionName = definitionName
        self.holder = holder
        self.condition = condition
        self.varDict = varDict
        self.jit = jit
        Clock.schedule_once(self.set_self)
        # .kv can't take data from another class during init. Everything needs to be init first
        # That's why the delay

    def set_self(self, dt):
        self.definition = vars(self.holder)[self.definitionName]
        self.text = self.definition

    # This is a significant change, we want to ensure confirmation
    """def _on_focus(self, instance, value, *largs):
        super(DataInput, self)._on_focus(instance, value, *largs)
        if value == False and self.text != self.copy:
            self.on_text_validate()"""

    def on_text_validate(self):
        # Enter
        try:
            self.definition = self.text
            self.func = createFunc(self.definition, self.varDict)
            # Try first to see if it can be converted sucessfully
        except Exception("Could not replace global variables properly"):
            TextPopup("Careful with Global Variables")
        except:
            TextPopup("Invalid Expression!\nRemember to multiply with *").open()
            return
        if self.conditionHolds(self.definition):
            vars(self.holder)[self.functionName] = jit(self.func) if self.jit else self.func
            vars(self.holder)[self.definitionName] = self.definition

    def conditionHolds(self, val):
        if self.condition == None: return True

        return True



# IMPORTANT
# IMPORTANT
# WARNING: eval() https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
# WARNING
# This creates a GLOBAL function. This is why it's allowed to return
# it seems locals can't change, but globals can
# https://stackoverflow.com/questions/41100196/exec-not-working-inside-function-python3-x
def createFunc(expression, variableDict):
    try:
        expressionFormated = expression.format(**variableDict)
    except:
        raise Exception("Could not replace global variables properly")
    exec("""
def funcManualGLOBAL(x, y, t=0., extra_param=None):
    return {}""".format(expressionFormated), globals())
    return funcManualGLOBAL

class WindowManager(ScreenManager):
    pass

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)

class examplesScreen(BoxLayout):
    pass


class SandboxScreen(Screen):
    paused = True

    settingsButton = ObjectProperty(None)


    def __init__(self, **kwargs):
        self._first_init()
        super(SandboxScreen, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def on_enter(self, *args):
        self.animation.reset_plot()

    def _first_init(self):
        self.Nx = 200; self.Ny = 200; L = 10.
        self.x0, self.y0 = -L, -L
        self.xf, self.yf = L, L

        """actualNx = Nx; actualNy = Ny; actualL = L
        actualx0, actualy0 = x0, y0
        actualxf, actualyf = xf, yf"""

        # We allow 16 global variables, which can be named
        self.nVar = 16
        self.extra_param = np.zeros(self.nVar, dtype=np.float64)
        self.extra_param[self.nVar-2] = -5.
        self.extra_param[self.nVar-1] = -2.5
        # A name can be assigned to each of these
        self.paramNames = [""]*(self.nVar-2) + ["px"] + ["py"]
        self.variablesDict = {'px': 'extra_param[{}]'.format(self.nVar-2), 'py': 'extra_param[{}]'.format(self.nVar-1)}

        self.initState = mathPhysics.gaussian2D(7, 1., -5., 7., 1., -2.5)
        self.initStateDef = \
            "1/(2*np.pi)**0.5 * np.exp(-1./4. * ((x-7)**2 + (y-7)**2)) * np.exp(1j * ({px}*x + {py}*y))"

        self.potential = mathPhysics.potentialBarrier
        self.potentialDef = "np.exp(-(x ** 2) / 0.1) * 5 / np.sqrt(0.1 * np.pi)"

        self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf, self.initState,
                                                   potential=self.potential, extra_param=self.extra_param)

        self.animation = animate.QuantumAnimation(self.QSystem, dtSim=0.01,
                                                  dtAnim=0.04, debugTime=True,
                                                  showPotential=True, updatePotential=True,
                                                  showMomentum=True, showEnergy=True, showNorm=True,
                                                  scalePsi=True, scaleMom=True, isKivy=True)

    def _finish_init(self, dt):
        self.plotBox = self.ids.plot
        self.plotBox.add_widget(self.animation.fig.canvas)

        self.ids.renorm.bind(on_release = self.renorm)

        #self.settingsButton.bind(on_release = self.dropdown.open)

    def renorm(self, dt):
        self.animation.QSystem.renorm()

    def stopPlaying(self):
        try: self.schedule.cancel()
        except: pass
        self.paused = True
        self.animation.paused = True

    def startPlaying(self):
        self.schedule = Clock.schedule_interval(self.play, self.animation.dtAnim)
        self.paused = False
        self.animation.paused = False

    def play(self, dt):
        if self.animation.paused:
            self.paused = self.animation.paused
            self.stopPlaying()
        else:
            self.animation.manualUpdate()
            #self.animation.update(self.animation.frame)
            #self.animation.frame += 1
            #self.animation.fig.canvas.draw()

    def setVarDict(self):
        self.variablesDict = \
            {self.paramNames[i]: "extra_param[{}]".format(i) for i in range(self.nVar) if self.paramNames[i] != ""}

        #print("Variables: ", self.variablesDict)

    def newSystem(self):
        self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf, self.initState,
                                                   potential=self.potential, extra_param=self.extra_param)

        self.animation.resetSystem(self.QSystem)
        #self.animation.reset_lists()
        #self.animation.reset_plot()

class TextPopup(Popup):
    def __init__(self, text, **kwargs):
        super(TextPopup, self).__init__(**kwargs)
        self.add_widget(Label(text=text))

class ColoredLabel(Label):
    pass

class SaveGifPopup(Popup):
    def __init__(self, window, duration=5., fileName="resultat", animwidth=12., animheight=7., **kwargs):
        self.window = window  # Window holds information such as QuantumSystem and Animation
        self.duration = duration
        self.fileName = fileName
        self.animwidth = animwidth
        self.animheight = animheight
        super(SaveGifPopup, self).__init__(**kwargs)

    def saveAnimation(self, fName, duration, type):
        anim = self.window.animation
        animationToSave = animate.QuantumAnimation(
            anim.QSystem, dtSim=anim.dtSim, stepsPerFrame=anim.stepsPerFrame, width=self.animwidth, height=self.animheight,
            duration=duration, dtAnim=anim.dtAnim, callbackProgress=True,
            showPotential=True, updatePotential=True,
            showMomentum=anim.showMomentum, showEnergy=anim.showMomentum, showNorm=anim.showNorm,
            scalePsi=anim.scalePsi, scaleMom=anim.scaleMom, isKivy=False)
        animationToSave.saveAnimation(fName, type)

    """def on_open(self):
        self.ids.Nx.text = str(self.window.animation.QSystem.Nx)
        self.ids.Ny.text = str(self.window.animation.QSystem.Ny)
        self.ids.x0.text = str(self.window.animation.QSystem.x0)
        self.ids.xf.text = str(self.window.animation.QSystem.xf)
        self.ids.y0.text = str(self.window.animation.QSystem.y0)
        self.ids.yf.text = str(self.window.animation.QSystem.yf)

        self.ids.t.text = str(self.window.animation.QSystem.t)

        self.ids.dtSim.text = str(self.window.animation.dtSim)"""

class ParametersPopup(Popup):
    def __init__(self, window, **kwargs):
        self.window = window  # Window holds information such as QuantumSystem and Animation
        super(ParametersPopup, self).__init__(**kwargs)

    """def on_open(self):
        self.ids.Nx.text = str(self.window.animation.QSystem.Nx)
        self.ids.Ny.text = str(self.window.animation.QSystem.Ny)
        self.ids.x0.text = str(self.window.animation.QSystem.x0)
        self.ids.xf.text = str(self.window.animation.QSystem.xf)
        self.ids.y0.text = str(self.window.animation.QSystem.y0)
        self.ids.yf.text = str(self.window.animation.QSystem.yf)

        self.ids.t.text = str(self.window.animation.QSystem.t)

        self.ids.dtSim.text = str(self.window.animation.dtSim)"""





class quantumMovementApp(App):
    def build(self):
        return WindowManager()
        #return kv


if __name__ == "__main__":
    quantumMovementApp().run()