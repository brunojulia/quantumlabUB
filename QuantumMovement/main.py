import matplotlib

from functools import partial

import random

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
from kivy.properties import NumericProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.factory import Factory

#matplotlib.use('module://kivy.garden.matplotlib.backend_kivyagg')
# Should this be used? Can't save to file then?

import numpy as np
from numba import jit, njit
import numba
import crankNicolson.crankNicolson2D as mathPhysics
from crankNicolson.crankNicolson2D import hred
import crankNicolson.animate as animate
import matplotlib.pyplot as plt
import time

####
# https://stackoverflow.com/questions/70629758/kivy-how-to-pass-arguments-to-a-widget-class
#####
"""
class ClassicalParticle(Widget):
    x = NumericProperty(0.)
    y = NumericProperty(0.)
    px = NumericProperty(0.)
    py = NumericProperty(0.)
    def __init__(self, QSystem, **kwargs):
        super(ClassicalParticle, self).__init__(**kwargs)
        self.QSystem = QSystem
        self.time = self.QSystem.t

        self.x = self.QSystem.expectedX()
        self.y = self.QSystem.expectedY()

        self.px = self.QSystem.expectedPX()
        self.py = self.QSystem.expectedPY()


    def move(self):"""



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


class SavedStatesPopup(Popup):
    def __init__(self, window, **kwargs):
        self.window = window
        super(SavedStatesPopup, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        # Lambda in loops! Careful with using iterating variable
        # Same with function, using global variable instead of value during creation
        # https://stackoverflow.com/questions/19837486/lambda-in-a-loop
        grid = self.ids.states
        #self.parts = []

        for state in self.window.savedStates:

            lbl = Label(text=state["name"])#,
            btnprev = Button(text="Previsualitza",
                             on_release = lambda x, state=state: PlotPopup(state).open())
            btnchan = Button(text="Canvia\na aquest",
                             on_release=lambda x, state=state: self.window.setState(state))
            btnsub = Button(text="Substreu\na l'actual",
                            on_release=lambda x, state=state: self.window.substractComponent(state))


            btndel = Button(text="Elimina", background_color=(0.6,0,0,0.8))

            def removeBind(*args, state=state,
                           lbl=lbl, btnprev=btnprev, btnchan=btnchan, btnsub=btnsub, btndel=btndel):
                #print(state["name"])
                for j in range(len(self.window.savedStates)):
                    if self.window.savedStates[j]["name"]==state["name"]:
                        del self.window.savedStates[j]
                #self.window.savedStates.remove(state)
                grid.remove_widget(lbl)
                grid.remove_widget(btnprev)
                grid.remove_widget(btnchan)
                grid.remove_widget(btnsub)
                grid.remove_widget(btndel)
                self.window.ids.stateName.text = "estat{}".format(len(self.window.savedStates))

            btndel.bind(on_release=removeBind)

            grid.add_widget(lbl)
            grid.add_widget(btnprev)
            grid.add_widget(btnchan)
            grid.add_widget(btnsub)
            grid.add_widget(btndel)


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
            if not (float(left) <= val <= float(right)):
            #if val < float(left) or float(right) < val:
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

################################################################################################################
#--------------------------------------------------------------------------------------------------------------#
#                           FUNCIONS EN GENERAL                                                             #

from crankNicolson.crankNicolson2D import gaussianPacket
"""
def gaussianPacket(x, x0, sigma, p0, extra_param=None):
    global hred
    return 1./(2*np.pi*sigma**2)**(0.25) * np.exp(-1./4. * ((x-x0)/sigma)**2) * np.exp(1j/hred * p0*(x))
"""


@jit#(cache=True)
def hssuau(x, k=1.):
    """
    Heavyside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation, less smooth
    """
    return 1./(1. + np.exp(-2.*k*x))


# IMPORTANT
# IMPORTANT
# WARNING: eval() https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
# WARNING
# This creates a GLOBAL function. This is why it's allowed to return
# it seems locals can't change, but globals can
# https://stackoverflow.com/questions/41100196/exec-not-working-inside-function-python3-x
from numpy import sin, cos, tan, arcsin, arccos, arctan, hypot, arctan2, degrees, radians, deg2rad, rad2deg,\
                  sinh, cosh, tanh, arcsinh, arccosh, arctanh,\
                  around, rint, fix, floor, ceil, trunc,\
                  exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2,\
                  power, sqrt, i0, sinc,\
                  sign, heaviside,\
                  pi

def createFunc(expression, variableDict):
    try:
        expressionFormated = expression.format(**variableDict)
    except:
        raise Exception("Could not replace global variables properly")
    exec("""
def funcManualGLOBAL(x, y, t=0., extra_param=np.array([])):
    r = sqrt(x*x + y*y)
    return {}""".format(expressionFormated), globals())
    return funcManualGLOBAL

class WindowManager(ScreenManager):
    pass

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)


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
        self.extra_param[0] = 0.
        self.extra_param[self.nVar-2] = -5.
        self.extra_param[self.nVar-1] = -2.5
        # A name can be assigned to each of these
        self.paramNames = ["Vx"] + [""]*(self.nVar-3) + ["px"] + ["py"]
        self.setVarDict()
        #self.variablesDict = {'px': 'extra_param[{}]'.format(self.nVar-2), 'py': 'extra_param[{}]'.format(self.nVar-1)}

        self.initState = mathPhysics.gaussian2D(7, 1., self.extra_param[self.nVar-2],
                                                7., 1., self.extra_param[self.nVar-1])
        self.initStateDef = \
            "gaussianPacket(x, 7, 1, {px}) * gaussianPacket(y, 7, 1, {py})"
            #"1/(2*pi)**0.5 * exp(-1./4. * ((x-7)**2 + (y-7)**2)) * exp(1j * ({px}*x + {py}*y))"

        self.potential = mathPhysics.potentialBarrier
        self.potentialDef = "exp(-(x ** 2) / 0.1) * 5 / sqrt(0.1 * pi)"

        self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf, self.initState,
                                                   potential=self.potential, extra_param=self.extra_param)

        self.animation = animate.QuantumAnimation(self.QSystem, dtSim=0.01,
                                                  dtAnim=0.05, debugTime=True,
                                                  showPotential=True, updatePotential=True,
                                                  showMomentum=True, showEnergy=True, showNorm=True,
                                                  scalePsi=True, scaleMom=True, isKivy=True, drawClassical=True)

        self.savedStates = []

        self.tempState = {"psi": self.QSystem.psi, "x0": self.QSystem.x0, "xf": self.QSystem.xf
                             ,                        "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                             "name": "temp"}

    def _finish_init(self, dt):
        self.plotBox = self.ids.plot
        self.plotBox.add_widget(self.animation.fig.canvas)

        self.ids.renorm.bind(on_release = self.renorm)

        #self.settingsButton.bind(on_release = self.dropdown.open)

    def renorm(self, dt):
        self.animation.QSystem.renorm()

    def saveState(self):
        repeated = False
        for state in self.savedStates:
            if self.ids.stateName.text == state["name"]: repeated = True

        if repeated:
            TextPopup("Nom ja fet servir!").open()

        else:
            self.savedStates.append({"psi": self.QSystem.psi.copy(), "x0": self.QSystem.x0, "xf": self.QSystem.xf
                                     ,                               "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                                     "name": self.ids.stateName.text})
            self.ids.stateName.text = "estat{}".format(len(self.savedStates))

    def setState(self, state):
        self.QSystem.setState(state)
        self.animation.reset_plot()

    def substractComponent(self, state):
        self.QSystem.substractComponent(state)
        self.animation.reset_plot()

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
        prevState = self.QSystem.psi
        x0Old = self.QSystem.x0
        xfOld = self.QSystem.xf
        y0Old = self.QSystem.y0
        yfOld = self.QSystem.yf
        self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf, prevState,
                                                   potential=self.potential, extra_param=self.extra_param,
                                                   x0Old=x0Old, xfOld=xfOld, y0Old=y0Old, yfOld=yfOld)

        self.animation.resetSystem(self.QSystem)

        #self.savedStates.clear()
        self.tempState = {"psi": self.QSystem.psi, "x0": self.QSystem.x0, "xf": self.QSystem.xf
                          ,                        "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                          "name": "temp"}

        #self.animation.reset_lists()
        #self.animation.reset_plot()

class TextPopup(Popup):
    def __init__(self, text, **kwargs):
        super(TextPopup, self).__init__(**kwargs)
        self.add_widget(Label(text=text))

class PlotPopup(Popup):
    def __init__(self, data, **kwargs):
        super(PlotPopup, self).__init__(**kwargs)
        self.data = data
        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        self.plotBox = self.ids.plot

        self.fig = plt.figure()
        FigureCanvasKivyAgg(self.fig)
        self.ax = self.fig.add_subplot()

        if self.data["psi"].dtype == np.complex128:
            self.psiMod = np.empty((len(self.data["psi"]), len(self.data["psi"][0])), dtype=np.float64)
            self.psiMod[:,:] = mathPhysics.abs2(self.data["psi"])
        else:
            self.psiMod = self.data["psi"]

        self.ax.imshow(self.psiMod.T, origin='lower',
                       extent=(self.data["x0"], self.data["xf"], self.data["y0"], self.data["yf"]),
                       aspect = 'equal', cmap = "viridis")

        self.plotBox.add_widget(self.fig.canvas)
        self.fig.canvas.draw()


class ExamplesScreen(Screen):
    def __init__(self, **kwargs):
        super(ExamplesScreen, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        # WALL
        definition = None  # default
        self.ids.exampselect.add_widget(
            Button(text="Barrera", on_release=partial(self.switch, definition=definition)))

        # GRAVITY
        definition = {"initState": mathPhysics.gaussian2D(0., 1., 0.,
                                                7., 1., 0.), "potential": mathPhysics.potentialGravity}
        self.ids.exampselect.add_widget(
            Button(text="Gravetat", on_release=partial(self.switch, definition=definition)))

        # UNCERTAINTY
        @jit
        def potentialClosingSoft(x, y, t, extra_param):
            global L
            # Heavyside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation
            r = np.sqrt(x * x + y * y)
            k = 5
            return 100 * 1 / (1 + np.exp(-2 * k * (r - 10. / 2 + 9 / 2 * (1 - 1. / (1 + 0.2 * t)))))
        definition = {"initState": mathPhysics.gaussian2D(0., 1., 0.,
                                                          0., 1., 0.), "potential": potentialClosingSoft,
                      "drawClassical":False, "showEnergy":False, "showNorm":False}
        self.ids.exampselect.add_widget(
            Button(text="Principi d'incertesa", on_release=partial(self.switch, definition=definition)))


    def switch(self, *args, definition=None):
        self.manager.get_screen("playscreen").set_self(definition)
        self.manager.transition.direction = "left"
        self.manager.current = "playscreen"

class GamesScreen(Screen):
    def __init__(self, **kwargs):
        super(GamesScreen, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        # Move the particle!
        kHarm = 8.
        extra_param = np.array([0., 0., 0., 0., 0.])

        @njit([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64[:])])
        def potentialHarmonicWellMoving(x, y, t, extra_param):
            res = 1 / 2. * kHarm * ((x - extra_param[0] - extra_param[2] * (t - extra_param[4])) ** 2
                                    + (y - extra_param[1] - extra_param[3] * (t - extra_param[4])) ** 2
                                    )
            if res > 100.: return 100.
            return res
        inicial2D = mathPhysics.eigenvectorHarmonic2DGenerator(0., 2, 0., 2, kHarm)
        QSystem = mathPhysics.QuantumSystem2D(initState=inicial2D,
                                              potential=potentialHarmonicWellMoving,
                                              extra_param=extra_param)
        def moveHarmonicWellKeyboard(event):
            if event.key == 'up' or 'down' or 'left' or 'right':
                t = QSystem.t
                extra_param[0] = extra_param[0] + extra_param[2] * (t - extra_param[4])
                extra_param[1] = extra_param[1] + extra_param[3] * (t - extra_param[4])
                extra_param[4] = t
            if event.key == 'up':
                extra_param[3] += 0.25
            if event.key == 'down':
                extra_param[3] -= 0.25
            if event.key == 'left':
                extra_param[2] -= 0.25
            if event.key == 'right':
                extra_param[2] += 0.25

        goalRadius = 2.
        goalX = QSystem.x0 + goalRadius + random.random() * (QSystem.xf - QSystem.x0 - 2 * goalRadius)
        goalY = QSystem.y0 + goalRadius + random.random() * (QSystem.yf - QSystem.y0 - 2 * goalRadius)
        global goalCircle; goalCircle = plt.Circle((goalX, goalY), goalRadius, alpha=0.2, color='black')
        global firstDraw; firstDraw = True
        global drawnCircle; drawnCircle = None

        def drawCircle(instance=None):
            global goalCircle, firstDraw, drawnCircle
            if firstDraw:
                drawnCircle = instance.axPsi.add_patch(goalCircle)
                firstDraw = False
            else:
                return drawnCircle

        definition = {
            "QSystem": QSystem,
            "extra_param": extra_param,
            "drawClassical": False, "duration": 10.,
            "extraCommands": [('key_press_event', moveHarmonicWellKeyboard)],
            "extraUpdates": [drawCircle],
            "showNorm": False, "showEnergy":False, "showMomentum":False
        }
        """"initState": mathPhysics.eigenvectorHarmonic2DGenerator(0., 2, 0., 2, 8.),
            "potential": potentialHarmonicWellMoving
        }"""
        self.ids.gameSelect.add_widget(
            Button(text="Transporta la partícula!", on_release=partial(self.switch, definition=definition)))


    def switch(self, *args, definition=None):
        self.manager.get_screen("playscreen").set_self(definition)
        self.manager.transition.direction = "left"
        self.manager.current = "playscreen"

class PlayScreen(Screen):
    def __init__(self, **kwargs):
        super(PlayScreen, self).__init__(**kwargs)

    def on_enter(self, *args):
        self.animation.reset_plot()

    def set_self(self, definition=None):
        if definition == None:
            definition = {}
        self.definition = definition

        self.paused = True
        self.Nx = 200; self.Ny = 200
        L = 10.
        self.x0, self.y0 = -L, -L
        self.xf, self.yf = L, L


        # We allow 16 global variables, which can be named
        self.nVar = 16
        self.extra_param = np.zeros(self.nVar, dtype=np.float64)
        self.extra_param[0] = 0.
        self.extra_param[self.nVar - 2] = -5.
        self.extra_param[self.nVar - 1] = -2.5
        # A name can be assigned to each of these
        self.paramNames = ["Vx"] + [""] * (self.nVar - 3) + ["px"] + ["py"]
        self.setVarDict()

        self.initState = mathPhysics.gaussian2D(7, 1., self.extra_param[self.nVar - 2],
                                                7., 1., self.extra_param[self.nVar - 1])
        self.initStateDef = \
            "gaussianPacket(x, 7, 1, {px}) * gaussianPacket(y, 7, 1, {py})"
        # "1/(2*pi)**0.5 * exp(-1./4. * ((x-7)**2 + (y-7)**2)) * exp(1j * ({px}*x + {py}*y))"

        self.potential = mathPhysics.potentialBarrier
        self.potentialDef = "exp(-(x ** 2) / 0.1) * 5 / sqrt(0.1 * pi)"

        for key in definition:
            vars(self)[key] = definition[key]

        if "QSystem" in self.definition:
            self.QSystem = self.definition["QSystem"]
        else:
            self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf,
                                                       self.initState,
                                                       potential=self.potential, extra_param=self.extra_param)

        animKeys = {"dtSim":0.01, "dtAnim":0.04, "debugTime":False, "duration":None,
                    "showPotential":True, "updatePotential":True, "showMomentum":True, "showEnergy":True, "showNorm":True,
                    "scalePsi":True, "scaleMom":True, "drawClassical":True,
                    "extraCommands":[], "extraUpdates":[]}
        for key in animKeys:
            animKeys[key] = definition.get(key, animKeys[key])

        self.animation = animate.QuantumAnimation(self.QSystem, **animKeys,
                                                  isKivy=True)

        self.savedStates = []

        self.tempState = {"psi": self.QSystem.psi, "x0": self.QSystem.x0, "xf": self.QSystem.xf
            , "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                          "name": "temp"}

        plotBox = BoxLayout(size_hint=(1, 0.8))
        plotBox.add_widget(self.animation.fig.canvas)


        buttonBox = BoxLayout(size_hint=(1, 0.2), orientation="horizontal", padding=10, spacing=20)

        resetButton = Button(text="Reset", on_press=self.resetAll)
        buttonBox.add_widget(resetButton)

        playButton = Button(text="Play/Pause", on_press= lambda x: self.startPlaying() if self.paused else self.stopPlaying())
        buttonBox.add_widget(playButton)

        def goBack(*args):
            self.manager.transition.direction = "right"
            self.manager.current = "examples"

        returnButton = Button(text="Retorna enrere", on_press = goBack)
        buttonBox.add_widget(returnButton)

        mainBox = BoxLayout(orientation="vertical")

        mainBox.add_widget(plotBox)
        mainBox.add_widget(buttonBox)
        self.add_widget(mainBox)

    def on_leave(self):
        self.stopPlaying()
        self.clear_widgets()

    def resetAll(self, *args):
        self.stopPlaying()
        self.clear_widgets()
        self.set_self(self.definition)

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

    def setVarDict(self):
        self.variablesDict = \
            {self.paramNames[i]: "extra_param[{}]".format(i) for i in range(self.nVar) if self.paramNames[i] != ""}





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
            scalePsi=anim.scalePsi, scaleMom=anim.scaleMom, isKivy=False, drawClassical=anim.drawClassical)
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