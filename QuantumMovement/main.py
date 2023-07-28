#import matplotlib
#matplotlib.use('agg')
#matplotlib.use('module://kivy.garden.matplotlib.backend_kivyagg')
# Should this be used? Can't save to file then? Better not, implicitly draws figure using agg

from functools import partial

import random

import kivy
from kivy.app import App
from kivy.uix.label import Label

from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.core.window import Window


from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from crankNicolson.animate import FigureCanvasKivyAggModified
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.factory import Factory

import numpy as np
from numba import jit, njit
import numba
import crankNicolson.crankNicolson2D as mathPhysics
from crankNicolson.crankNicolson2D import hred
import crankNicolson.animate as animate
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 12})
import time

####
# https://stackoverflow.com/questions/70629758/kivy-how-to-pass-arguments-to-a-widget-class
#####

#import warnings     # For debugging. Warnings stop the program, and thus also show where they occur
#warnings.filterwarnings('error')


unit_dist = '2 Å'
unit_time = '1/3 fs'
unit_energy = '2 eV'
unit_mom = r'$\frac{1}{2}\hbar Å^{-1}$'#'1/3 eV·fs/Å'  #'2 eV · 1/3 fs / 2 Å'
# hred = 2 eV · 1/3 fs = 2/3 eV · fs
# ℏ ℏ ℏ ℏ


class PlayButton(ToggleButton):
    pass

class BoolCheckBox(CheckBox):
    pass

class FastGraphicsCheckbox(CheckBox):
    def on_active(self, *args):
        animate.optimizeGraphics = self.active




class GlobalVariable(GridLayout):
    def __init__(self, source, names, sliders, num=0, **kwargs):
        super(GlobalVariable, self).__init__(cols=4,**kwargs)
        self.num = num
        self.names = names
        self.sliders = sliders # List of: show variable i as a slider?
        self.source = source # List of variables, numpy array

        self.label = Label(text="Var.\n{}".format(num), size_hint_x=0.2)

        self.nameText = DataInput(attribute="names", index=self.num, holder=self, condition="unique", multiline=False, size_hint_x=0.3)

        self.valText = DataInput(attribute="source", index=self.num, holder=self, multiline=False, size_hint_x=0.3)

        self.sliderQuery = GridLayout(rows=2, size_hint_x=0.2)
        self.sliderQuery.add_widget(Label(text="Slider?"))
        sliderCheck = CheckBox(active=sliders[num])
        def updateSlider(checkbox, active):
            sliders[num] = active
        sliderCheck.bind(active=updateSlider)
        self.sliderQuery.add_widget(sliderCheck)



        #self.layout = GridLayout(cols=3, size=(100, 100))
        self.add_widget(self.label)
        self.add_widget(self.nameText)
        self.add_widget(self.valText)
        self.add_widget(self.sliderQuery)
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
            self.layout.add_widget(GlobalVariable(self.window.extra_param, self.window.paramNames, self.window.paramSliders, num=i))
        self.add_widget(self.layout)

    def on_dismiss(self):
        super(GlobalVariablesPopup, self).on_dismiss()
        # We need to wait to make sure all DataInputs finish on unfocus
        def updateStuff(*param):
            self.window.setVarDict()
            self.window.setSliders()
        Clock.schedule_once(updateStuff)


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
                        break
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

class SavedEigenstatesPopup(Popup):
    def __init__(self, window, **kwargs):
        self.window = window
        self.tol = 1e-4
        self.maxiter = 20
        super(SavedEigenstatesPopup, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):

        """def updateProgress(val):
            self.ids.progress.value = val*100
        self.callback = updateProgress"""

        # Lambda in loops! Careful with using iterating variable
        # Same with function, using global variable instead of value during creation
        # https://stackoverflow.com/questions/19837486/lambda-in-a-loop
        grid = self.ids.states
        #self.parts = []

        count = 0
        prev = None
        for E, eigen in self.window.QSystem.eigenstates:
            self.window.tempState["psi"] = eigen
            state = self.window.tempState

            if prev == E: count+=1
            lbl = Label(text="E={:.4e} |{}".format(E, count))#,
            prev = E

            btnprev = Button(text="Previsualitza",
                             on_release = lambda x, state=state.copy(): PlotPopup(state).open())
            btnchan = Button(text="Canvia\na aquest",
                             on_release=lambda x, state=state.copy(): self.window.setState(state))
            btnsub = Button(text="Substreu\na l'actual",
                            on_release=lambda x, state=state.copy(): self.window.substractComponent(state))

            # Bad, we are multiplying memory use here... We should not create more copies of states for each button

            btndel = Button(text="Elimina", background_color=(0.6,0,0,0.8))

            def removeBind(*args, state=state,
                           lbl=lbl, btnprev=btnprev, btnchan=btnchan, btnsub=btnsub, btndel=btndel, E=E, count=count):
                #print(state["name"])
                rep = 0
                for j in range(len(self.window.QSystem.eigenstates)):
                    if self.window.QSystem.eigenstates[j][0]==E:
                        if rep == count:
                            del self.window.QSystem.eigenstates[j]
                            break
                        else: rep+=1

                grid.remove_widget(lbl)
                grid.remove_widget(btnprev)
                grid.remove_widget(btnchan)
                grid.remove_widget(btnsub)
                grid.remove_widget(btndel)

            btndel.bind(on_release=removeBind)

            grid.add_widget(lbl)
            grid.add_widget(btnprev)
            grid.add_widget(btnchan)
            grid.add_widget(btnsub)
            grid.add_widget(btndel)
    def eigenFind(self):
        self.count = 0
        self.window.QSystem.setState(mathPhysics.func1)
        def eigenLoop(*args):
            if self.count < self.maxiter:
                if self.window.QSystem.approximateEigenstate(tol=self.tol, maxiter=5, resetInit=False):
                    self.window.animation.manualUpdate(onlyDraw=True)
                    self.dismiss()
                    return
                else:
                    Clock.schedule_once(eigenLoop, 0.)
                self.count+=1
                self.ids.progress.value = self.count/self.maxiter * 100
            else:
                TextPopup("No s'ha pogut trobar!").open()
                self.window.animation.manualUpdate(onlyDraw=True)
                self.dismiss()
        Clock.schedule_once(eigenLoop)





class DataInput(TextInput):
    """
    Holds an attribute which it can change/modify. The attribute is passed by value,
    so to modify it in the class instance it refers to that class instance needs to be passed.
    Then it can be modified with getattr(class/instance, attribute) and setattr     ## NOT GOOD PRACTICE before: vars(classInstance) / classInstance.__dict__.
    Which returns a dictionary of variables, mutable.
    """
    text_width = NumericProperty()
    def __init__(self, attribute = None, index=None, holder = None, condition = None, callback=None, centered=False, scientific=False, maxDecimal=6, **kwargs):
        self.scientific = False
        self.maxDecimal = maxDecimal
        self.centered = centered

        super(DataInput, self).__init__(**kwargs)
        self.attribute = attribute
        self.index = index
        self.holder = holder
        self.condition = condition
        self.callback = callback

        Clock.schedule_once(self.set_self)
        # .kv can't take data from another class during init. Everything needs to be init first
        # That's why the delay

    def set_self(self, dt):
        # Attribute is set in kivy language
        #print(id(self.attribute)) Check
        self.attributeVal = getattr(self.holder,self.attribute) if self.index is None else \
                            getattr(self.holder,self.attribute)[self.index]
        self.type = type(self.attributeVal)
        form = "{:." + str(self.maxDecimal) + ('f' if not self.scientific else 'e') +'}'
        self.text = str(self.attributeVal) if self.type is not float else form.format(self.attributeVal).rstrip('0')
        self.copy = self.text

    def _on_focus(self, instance, value, *largs):
        # When unfocusing also apply changes. Value: false -> unfocus
        super(DataInput, self)._on_focus(instance, value, *largs)
        if value == False and self.text != self.copy:
            self.on_text_validate()


    def update_padding(self, *args):
        '''
        Update the padding so the text is centered
        '''
        self.text_width = self._get_text_width(self.text, self.tab_width, self._label_cached)


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
                setattr(self.holder, self.attribute, self.type(self.text))
            else:
                getattr(self.holder, self.attribute)[self.index] = self.type(self.text)  # Dangerous, look here
            self.copy = self.text
        else:
            self.text = self.copy
        if self.callback != None:
            self.callback(self.attributeVal)

    def conditionHolds(self, val):
        if self.condition == None: return True
        elif self.condition == "fixed":
            TextPopup("Fixed Value", title="Warning").open()
            return False
        elif self.condition == "notNothing":
            if val == "": return False
        elif self.condition == "unique" and self.index != None:
            ### Not too general
            ### We assume here no name is allowed to be repeated as a name
            if val != "" and val in getattr(self.holder,self.attribute):
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
        elif self.condition.startswith("gt"):
            if not (float(self.condition[2:]) < val):
                TextPopup("Must be greater than "+self.condition[2:], title="Warning").open()
                return False
        elif self.condition.startswith("geq"):
            if not (float(self.condition[2:]) <= val):
                TextPopup("Must be greater than or equal to "+self.condition[2:], title="Warning").open()
                return False
        elif self.condition.startswith("lt"):
            if not (val < float(self.condition[2:])):
                TextPopup("Must be less than "+self.condition[2:], title="Warning").open()
                return False
        elif self.condition.startswith("leq"):
            if not (val <= float(self.condition[2:])):
                TextPopup("Must be less than or equal to "+self.condition[2:], title="Warning").open()
                return False

        return True




"""
CURRENT PROBLEMS WITH THIS

Slider references on creation. So it's dangerous! For example, by default
playscreen only has a one dimensional array extra_param
if we create 2 sliders for the 2 parameters we want to modify this will
break as the slider will be created before the parameters are loaded. So we will
try to access extra_param[1] which is invalid
"""
class DataSlider(Slider):
    """
    Similar and linked to a DataInput, but user fiendly form of a Slider. Only numeric variables!
    """
    # By default step = 0, which means pixel resolution for changes
    def __init__(self, attribute = None, index=None, holder = None, callback=None, isPotential=True, **kwargs):
        self.attribute = attribute
        self.index = index
        self.holder = holder
        self.callback = callback
        self.isPotential = isPotential
        self.firstTime = True
        Clock.schedule_once(self.set_self)
        super(DataSlider, self).__init__(**kwargs)
        # .kv can't take data from another class during init. Everything needs to be init first
        # That's why the delay

    def set_self(self, dt):
        # Attribute is set in kivy language
        # print(id( self.attribute)) Check
        self.attributeVal = getattr(self.holder, self.attribute) if self.index is None else \
            getattr(self.holder,self.attribute)[self.index]

        self.text = str(self.attributeVal)
        self.copy = self.text

    def on_value(self, instance, val):
        # It gets fired when the slider is initalized! So things are not yet defined if not careful (super is called last)
        if self.index is None:
            setattr(self.holder, self.attribute, self.value)
        else:
            getattr(self.holder, self.attribute)[self.index] = self.value
        if self.callback is not None:
            self.callback(val)

        if not self.firstTime:
            if self.isPotential and self.holder.animation.paused: self.holder.animation.updatePotentialDraw()
        else: self.firstTime = not self.firstTime


class CustomDataSlider(BoxLayout):
    # Text showing value and name
    # Slider + custom slider ranges (max val and min val)

    ###### Maybe Clock once???? Need to check, maybe clock_once already inside slider/datainput is enough
    def __init__(self, name = None, attribute = None, index=None, holder = None, orientation="horizontal", min = -1., max = 1.,
                 isPotential=True, value=None, **kwargs):
        super(CustomDataSlider, self).__init__(orientation=orientation, **kwargs)

        if orientation == "horizontal":
            self.size_hint_x = 1
            self.size_hint_y = None
            self.height = 85
            sizeHintSlid = (0.7, 1)
        else:
            self.size_hint_x = None
            self.size_hint_y = 1
            self.width = 85
            sizeHintSlid = (1, 0.8)

        sizeHintText = (0.1, 1) if self.orientation == "horizontal" else (1, 0.1)
        self.name = name if name is not None else self.attribute
        self.label = Label(text="[b]" + self.name + "[/b]", size_hint=sizeHintText, markup=True)

        def updateLabel(newVal):
            self.label.text = "[b]" + self.name + "[/b] =\n{:.2f}".format(newVal)

        self.slider = DataSlider(attribute, index, holder, value=max if value is None else value, min=min, max=max, size_hint=sizeHintSlid,
                                 orientation=orientation, callback=updateLabel, isPotential=isPotential)
        self.name = name
        self.attribute = attribute
        self.orientation = orientation
        Clock.schedule_once(self._finish_init)

    def _finish_init(self, dt):
        sizeHintText = (0.1, 1) if self.orientation == "horizontal" else (1, 0.1)
        self.minDat = DataInput(attribute="min", holder=self.slider, condition="lt{0}".format(self.slider.max),
                                size_hint=sizeHintText, centered=True)

        def updateMin(newMax):
            self.minDat.condition = "lt{0}".format(newMax)


        self.maxDat = DataInput(attribute="max", holder=self.slider, condition="gt{0}".format(self.slider.min),
                                size_hint=sizeHintText, centered=True)

        def updateMax(newMin):
            self.maxDat.condition = "gt{0}".format(newMin)

        self.maxDat.callback = updateMin
        self.minDat.callback = updateMax

        # Layout  [min] [----- slider -----] [max]
        #self.layout = BoxLayout(orientation=self.orientation)
        self.add_widget(self.label)
        if self.orientation == "horizontal":
            self.add_widget(self.minDat)
            self.add_widget(self.slider)
            self.add_widget(self.maxDat)
        else:
            self.add_widget(self.maxDat)
            self.add_widget(self.slider)
            self.add_widget(self.minDat)
        #self.add_widget(self.layout)



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
        self.definition = getattr(self.holder, self.definitionName)
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
            # If it can't create the function, it should not alter the values, it should fail here
            if self.conditionHolds(self.definition):
                setattr(self.holder, self.functionName, jit(self.func) if self.jit else self.func)
                setattr(self.holder, self.definitionName, self.definition)

            return 0 # Returns True when everything is OK
        except MaliciousInput:
            exit(print("ALERTA: AIXÒ NO ES POT FER"))
        except InvalidFormat:
            TextPopup("Compte amb les variables globals").open()
        except:
            TextPopup("Expressió Invàlida!\nRecorda multiplicar amb *\nI compte amb divisions per 0\nPer 1/r posa 1/(r+numPetit)").open()


    def conditionHolds(self, val):
        if self.condition == None: return True

        return True

################################################################################################################
#--------------------------------------------------------------------------------------------------------------#
#                           FUNCIONS EN GENERAL                                                             #

from crankNicolson.crankNicolson2D import gaussianPacket
from crankNicolson.crankNicolson2D import eigenvectorsHarmonic1D
"""
def gaussianPacket(x, x0, sigma, p0, extra_param=None):
    global hred
    return 1./(2*np.pi*sigma**2)**(0.25) * np.exp(-1./4. * ((x-x0)/sigma)**2) * np.exp(1j/hred * p0*(x))
"""


@jit#(cache=True)
def heaviside(x, k=1.):
    """
    Heaviside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation, less smooth
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
                  sign,\
                  pi

ln = log #for commodity

# Heaviside is not supported in numba


class InvalidFormat(Exception):
    pass

class MaliciousInput(Exception):
    pass



def createFunc(expression, variableDict):
    if expression == "": raise InvalidFormat
    try:
        # Things like {px} are substituted by their corresponding actual extra_param
        expressionFormated = expression.format(**variableDict)
    except:
        raise InvalidFormat #Exception("Could not replace global variables properly")

    #################
    # SAFETY CHECKS #
    #################
    if "print" in expression or "import" in expression or "sys" in expression or "os." in expression or "open" in expression\
            or "__" in expression:# or "__builtins__" in expression:
        raise MaliciousInput
        # This is very very bad. Someone is doing something really wrong


    exec("""
def funcManualGLOBAL(x, y, t=0., extra_param=np.array([])):
    r = sqrt(x*x + y*y)
    return {}""".format(expressionFormated), globals())

    # After defining the function, we test it once, to see if it works. If not, it will raise an exception,
    # which should be catched where createFunc is used
    if not np.isfinite( funcManualGLOBAL(0., 0., 0., np.array([0.5]*100)) ):
        raise ZeroDivisionError
        # it can happen for other reasons too though.
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
        self.extra_param[1] = 5.
        self.extra_param[2] = 1.
        self.extra_param[self.nVar-2] = -5.
        self.extra_param[self.nVar-1] = -2.5
        # A name can be assigned to each of these
        self.paramNames = ["Vx"] + ["Vh"] + ["Vw"]+[""]*(self.nVar-5) + ["px"] + ["py"]
        self.paramSliders = [True] + [False] + [True] + [False]*(self.nVar-2)
        self.sliders = []
        self.setSliders(firstInit=True)
        self.variablesDict = {}
        self.setVarDict()
        #self.variablesDict = {'px': 'extra_param[{}]'.format(self.nVar-2), 'py': 'extra_param[{}]'.format(self.nVar-1)}

        self.initState = mathPhysics.gaussian2D(7, 1., self.extra_param[self.nVar-2],
                                                7., 1., self.extra_param[self.nVar-1])
        self.initStateDef = \
            "gaussianPacket(x, 7, 1, {px}) * gaussianPacket(y, 7, 1, {py})"
            #"1/(2*pi)**0.5 * exp(-1./4. * ((x-7)**2 + (y-7)**2)) * exp(1j * ({px}*x + {py}*y))"

        self.potential = mathPhysics.potentialBarrierYCustom
        self.potentialDef = "{Vh} * exp(-((x-{Vx}) ** 2) / 0.1) / sqrt(0.1 * pi) * (0 if abs(y) < {Vw} else 1)"

        self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf, self.initState,
                                                   potential=self.potential, extra_param=self.extra_param)

        self.animation = animate.QuantumAnimation(self.QSystem, dtSim=0.01,
                                                  dtAnim=0.05, debugTime=True,
                                                  showPotential=True, updatePotential=True,
                                                  showMomentum=True, showEnergy=True, forceEnergy=True, showNorm=True, forceNorm=True,
                                                  scalePsi=True, scaleMom=True, isKivy=True, drawClassical=True,
                                                  unit_dist=unit_dist, unit_time=unit_time, unit_energy=unit_energy, unit_mom=unit_mom)

        self.savedStates = []

        self.tempState = {"psi": self.QSystem.psi, "x0": self.QSystem.x0, "xf": self.QSystem.xf
                             ,                        "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                             "name": "temp"}

    def _finish_init(self, dt):
        self.plotBox = self.ids.plot
        self.plotBox.add_widget(self.animation.fig.canvas)

        for slider in self.sliders:
            self.plotBox.add_widget(slider)

        self.ids.renorm.bind(on_release = self.renorm)

        #self.settingsButton.bind(on_release = self.dropdown.open)

    def renorm(self, dt):
        """Manually Renormalize"""
        """# All the commented things are for testing, as renormalize is a harmless button by itself
        print("psi:",self.animation.axPsi.get_children())
        print("norm:",self.animation.axNorm.get_children())
        print("energy:",self.animation.axEnergy.get_children())
        print("momentum:",self.animation.axMomentum.get_children())"""

        """state = {"psi": self.QSystem.psiCopy, "x0": self.QSystem.x0, "xf": self.QSystem.xf
            , "y0": self.QSystem.y0, "yf": self.QSystem.yf,
         "name": "noname"}
        PlotPopup(state).open()"""

        print("pxMom:",self.QSystem.expectedPx(), ", pxPsi:",self.QSystem.expectedPxPsi())
        print("pyMom:", self.QSystem.expectedPy(), ", pyPsi:", self.QSystem.expectedPyPsi())

        self.QSystem.momentumSpace()
        print(mathPhysics.euclidNorm(self.QSystem.psiMom, self.QSystem.Px[1]-self.QSystem.Px[0], self.QSystem.Py[1]-self.QSystem.Py[0]))
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
        self.animation.manualUpdate(onlyDraw=True)
        #self.animation.reset_plot()

    def substractComponent(self, state):
        self.QSystem.substractComponent(state)
        self.animation.manualUpdate(onlyDraw=True)
        #self.animation.reset_plot()

    def stopPlaying(self):
        try: self.schedule.cancel()
        except: pass
        self.ids.pausePlay.state = 'normal'
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
        self.variablesDict.clear()
        for i in range(self.nVar):
            if self.paramNames[i] != "": self.variablesDict[self.paramNames[i]] = "extra_param[{}]".format(i)
        """#This modifies reference of dict. So it doesn't work if other people hold a copy
        self.variablesDict = \
            {self.paramNames[i]: "extra_param[{}]".format(i) for i in range(self.nVar) if self.paramNames[i] != ""}"""

        #print("Variables: ", self.variablesDict)

    def setSliders(self, firstInit = False):
        if not firstInit:
            for slider in self.sliders:
                self.plotBox.remove_widget(slider)

        self.sliders.clear()
        for i in range(self.nVar):
            if self.paramSliders[i]:
                self.sliders.append(CustomDataSlider(name=self.paramNames[i], attribute="extra_param", index=i, holder=self,
                                                     orientation="vertical", value=float(self.extra_param[i]),
                                                     min=float(self.extra_param[i]-10.), max=float(self.extra_param[i]+10.)))
                # Careful with numpy, weird interaction with NumericProperty:
                # https://kivy.org/doc/stable/api-kivy.properties.html#kivy.properties.NumericProperty

        if not firstInit:
            for slider in self.sliders:
                self.plotBox.add_widget(slider)

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

        self.fig, self.ax = plt.subplots()
        FigureCanvasKivyAgg(self.fig)
        self.ax = self.fig.gca()  # Just in case?
        #self.ax = self.fig.add_subplot()   # Maybe this doesn't always work

        if self.data["psi"].dtype == np.complex128:
            self.psiMod = np.empty((len(self.data["psi"]), len(self.data["psi"][0])), dtype=np.float64)
            self.psiMod[:,:] = mathPhysics.abs2(self.data["psi"])
        else:
            self.psiMod = self.data["psi"]

        # Didn't work on some other computers? ax not defined?
        datPlot = self.ax.imshow(self.psiMod.T, origin='lower',
                       extent=(self.data["x0"], self.data["xf"], self.data["y0"], self.data["yf"]),
                       aspect = 'equal', cmap = "viridis")

        self.fig.colorbar(datPlot, ax=self.ax, label=self.data.get("unit_col",r'$Å^{-2}$'))

        self.ax.set_xlabel("x ({})".format(self.data.get("unit_ax",'Å')))
        self.ax.set_ylabel("y ({})".format(self.data.get("unit_ax", 'Å')))

        self.plotBox.add_widget(self.fig.canvas)
        self.fig.canvas.draw()

    def on_dismiss(self):
        #super().on_dismiss()
        plt.close(self.fig)



class ExamplesScreen(Screen):
    def __init__(self, **kwargs):
        super(ExamplesScreen, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        # ----------- ----------- ----------- ----------- -----------
        # WALL
        definition = None  # default
        self.ids.exampselect.add_widget(
            Button(text="Barrera", on_release=partial(self.switch, definition=definition)))

        # ----------- ----------- ----------- ----------- -----------


        # ----------- ----------- ----------- ----------- -----------
        # GRAVITY
        definition = {"initState": mathPhysics.gaussian2D(0., 1., 0.,
                                                7., 1., 0.), "potential": mathPhysics.potentialGravity}
        self.ids.exampselect.add_widget(
            Button(text="Gravetat", on_release=partial(self.switch, definition=definition)))

        # ----------- ----------- ----------- ----------- -----------

        # ----------- ----------- ----------- ----------- -----------
        # UNCERTAINTY
        @jit
        def potentialClosingSoft(x, y, t, extra_param):
            global L
            # Heaviside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation
            r = np.sqrt(x * x + y * y)
            k = 1
            return 100 * 1 / (1 + np.exp(-2 * k * (r - 10. / 2 + 9.5 / 2 * (1 - 1. / (1 + 0.2 * t))))) # 0.5 originally


        def plotUnc(args):
            grid = BoxLayout(size_hint_x=0.3, size_hint_y=0.4, padding=0, pos_hint={'center_x': .5, 'center_y': .5})
            grid.add_widget(args["canvasFig"])
            return grid

        def setUncertainty(args):
            args["figUnc"] = plt.figure()
            args["axUnc"] = plt.axes()
            args["canvasFig"] = FigureCanvasKivyAggModified(args["figUnc"])
            XUncMin = 1 / np.linspace(0.5,10., 10) # better spacing
            YUncMin = mathPhysics.hred/2 / XUncMin

            args["axUnc"].set(xlabel=r'$\sigma_x$ ({})'.format(unit_dist), ylabel=r'$\sigma_{{p_x}}$ ({})'.format(unit_mom),
                              title='Heisenberg: Relació entre incerteses\n(variàncies $\\sigma_{p_x}$ i $\\sigma_x$)')
            args["axUnc"].plot(XUncMin, YUncMin, 'r--', label=r'$\sigma_x \sigma_{p_x} = \hbar/2$')
            args["axUnc"].grid()
            args["axUnc"].legend()

            args["axUnc"].set_xscale('log')
            args["axUnc"].set_yscale('log')

            args["datUnc"], = args["axUnc"].plot(np.array([1.]),np.array([1./2.]))
            args["datUncPoint"], = args["axUnc"].plot(np.array([1.]), np.array([1. / 2.]), 'o')
            args["sigmax"] = []
            #args["sigmay"] = []
            args["sigmapx"] = []
            #args["sigmapy"] = []

        def extra_update_unc(args):
            ps = self.manager.get_screen("playscreen")
            varX = ps.QSystem.varX()
            args["sigmax"].append(varX)
            varPx = ps.QSystem.varPx()
            args["sigmapx"].append(varPx)

            args["datUnc"].set_data(args["sigmax"], args["sigmapx"])
            args["datUncPoint"].set_data(varX, varPx)

            #args["sigmay"].append(ps.QSystem.varY())
            #args["sigmapy"].append(ps.QSystem.varPy())

            """args["figUnc"].draw_artist(args["axUnc"].patch)
            args["figUnc"].draw_artist(args["datUnc"])
            args["figUnc"].draw_artist(args["datUncPoint"])"""
            args["canvasFig"].draw()

        def extra_on_enter_unc(args):
            args["figUnc"].tight_layout()
            args["canvasFig"].draw()




        def extra_clean_unc(args):
            plt.close(args["figUnc"])

        definition = {"initState": mathPhysics.gaussian2D(0., 1., 0.,
                                                          0., 1., 0.), "potential": potentialClosingSoft,
                      "drawClassical":False, "drawExpected": False, "showEnergy":False, "plotWidget": plotUnc,
                      "extra_update": extra_update_unc, "extra_clean": extra_clean_unc, "extra_on_enter": extra_on_enter_unc
                      }

        doubleUncertainty = GridLayout(rows=2)
        uncAutom = Button(text="Principi d'incertesa",
                          on_release=partial(self.switch, definition=definition, setExtraArgs=setUncertainty))


        @jit
        def potentialClosingManual(x, y, t, extra_param):
            global L
            # Heaviside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation
            r = np.sqrt(x * x + y * y)
            return 100 * heaviside(r-extra_param[0], 1.)

        def plotUncManual(args):
            box = BoxLayout(orientation='horizontal', size_hint_x=0.3)
            box.add_widget(plotUnc(args))
            box.add_widget(CustomDataSlider(name="R", attribute="extra_param", index=0, holder=self.manager.get_screen("playscreen"),
                                                     orientation="vertical", min=0., max=10., value=5.))
            return box

        definition = {"initState": mathPhysics.gaussian2D(0., 1., 0.,
                                                          0., 1., 0.), "potential": potentialClosingManual,
                      "drawClassical": False, "drawExpected": False, "showEnergy": False, "plotWidget": plotUncManual,
                      "extra_update": extra_update_unc, "extra_clean": extra_clean_unc,
                      "extra_on_enter": extra_on_enter_unc, 'extra_param':np.array([5.])
                      }


        uncManual = Button(text="Principi d'incertesa (manual)",
                          on_release=partial(self.switch, definition=definition, setExtraArgs=setUncertainty))

        doubleUncertainty.add_widget(uncAutom)
        doubleUncertainty.add_widget(uncManual)
        self.ids.exampselect.add_widget(doubleUncertainty)
        # ----------- ----------- ----------- ----------- -----------

        # ----------- ----------- ----------- ----------- -----------
        # EHRENFEST 2. HARMONIC OSCILLATOR
        definition = {"initState": mathPhysics.gaussian2D(0., 1., -3.,
                                                          3., 1., 0.),
                      "potential": mathPhysics.potentialHarmonic,
                      "extra_param": np.array([2.]),
                      "dtSim": 0.005, "scalePot":False,
                      "drawClassical":True, "drawClassicalTrace":True, "drawExpected":True, "drawExpectedTrace":True,
                      "plotWidget": CustomDataSlider(name="k", attribute="extra_param", index=0, holder=self.manager.get_screen("playscreen"),
                                                     orientation="vertical", min=1., max=2.)}
        self.ids.exampselect.add_widget(
            Button(text="Oscil·lador Harmònic", on_release=partial(self.switch, definition=definition)))

        # ----------- ----------- ----------- ----------- -----------

    def switch(self, *args, definition=None, setExtraArgs=None):
        self.manager.get_screen("playscreen").set_self(definition, setExtraArgs=setExtraArgs)
        self.manager.transition.direction = "left"
        self.manager.current = "playscreen"
        self.manager.get_screen("playscreen").sourceScreen = "examples"

class GamesScreen(Screen):
    def __init__(self, **kwargs):
        super(GamesScreen, self).__init__(**kwargs)

        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        # Move the particle!
        kHarm = 8.
        height = 100.
        extra_param = np.array([0., 0., 0., 0., 0., kHarm, height])

        @njit([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64[:])])
        def potentialHarmonicWellMovingSoft(x, y, t, extra_param):
            res = 1 / 2. * extra_param[5] * ((x - extra_param[0] - extra_param[2] * (t - extra_param[4])) ** 2
                                    + (y - extra_param[1] - extra_param[3] * (t - extra_param[4])) ** 2
                                    )
            if res > extra_param[6]: return extra_param[6]
            return res

        @njit([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64[:])])
        def potentialHarmonicWellMoving(x, y, t, extra_param):
            res = 1 / 2. * extra_param[5] * ((x - extra_param[2]) ** 2 + (y - extra_param[3]) ** 2)
            if res > extra_param[6]: return extra_param[6]
            return res
        inicial2D = mathPhysics.eigenvectorHarmonic2DGenerator(0., 2, 0., 2, kHarm)
        QSystem = mathPhysics.QuantumSystem2D(initState=inicial2D,
                                              potential=potentialHarmonicWellMoving,
                                              extra_param=extra_param)
        """def moveHarmonicWellKeyboard(event):
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
                extra_param[2] += 0.25"""

        """global goalRadius; goalRadius = 2.
        global score; score = 0
        global goalX; goalX = QSystem.x0 + goalRadius + random.random() * (QSystem.xf - QSystem.x0 - 2 * goalRadius)
        global goalY; goalY = QSystem.y0 + goalRadius + random.random() * (QSystem.yf - QSystem.y0 - 2 * goalRadius)
        global goalCircle; goalCircle = plt.Circle((goalX, goalY), goalRadius, alpha=0.2, color='black')
        global firstDraw; firstDraw = True
        global drawnCircle; drawnCircle = None"""

        def setMoveGame(args):
            ps = self.manager.get_screen("playscreen")
            args["kHarm"] = 8.
            args["height"] = 100.
            np.copyto(extra_param, np.array([0., 0., 0., 0., 0., kHarm, height]))
            args["goalRadius"] = 2.
            args["score"]=0
            args["lives"]=3
            args["goalX"] = ps.QSystem.x0 + args["goalRadius"] + random.random() * (ps.QSystem.xf - ps.QSystem.x0 - 2 * args["goalRadius"])
            args["goalY"] = ps.QSystem.y0 + args["goalRadius"] + random.random() * (ps.QSystem.yf - ps.QSystem.y0 - 2 * args["goalRadius"])
            args["goalCircle"] = plt.Circle((args["goalX"], args["goalY"]), args["goalRadius"], alpha=0.2, color='black')
            args["firstDraw"] = True
            args["drawnCircle"] = None

        ##### Three ideas for movement:
        #   - Soft movement. We change "speed" with keyboard
        #   - Direct movement. We change the position of the well directly
        #           - With keyboard
        #           - Directly with mouse!?


        def extra_keyboard_movegame(ps, keyboard, keycode, text, modifiers):
            if keycode[1] == 'up' or 'down' or 'left' or 'right':
                t = ps.QSystem.t
                ps.extra_param[0] = ps.extra_param[0] + ps.extra_param[2] * (t - ps.extra_param[4])
                ps.extra_param[1] = ps.extra_param[1] + ps.extra_param[3] * (t - ps.extra_param[4])
                ps.extra_param[4] = t
            if keycode[1] == 'up':
                ps.extra_param[3] += 0.25
            if keycode[1] == 'down':
                ps.extra_param[3] -= 0.25
            if keycode[1] == 'left':
                ps.extra_param[2] -= 0.25
            if keycode[1] == 'right':
                ps.extra_param[2] += 0.25
            return True



        def drawCircle(instance=None):
            ps = self.manager.get_screen("playscreen")
            if ps.extraArgs["firstDraw"]:
                instance.drawnCircle = instance.axPsi.add_patch(ps.extraArgs["goalCircle"])
                ps.extraArgs["firstDraw"] = False
                instance.observedParticle = None
            elif instance.frame%(6/0.04)==0: # Every 6 seconds
                instance.QSystem.modSquared()
                i, j = mathPhysics.generateAsDistribution(instance.QSystem.psiMod)
                x, y = instance.QSystem.X[i], instance.QSystem.X[j]

                instance.observedParticle = instance.axPsi.add_patch(plt.Circle((x, y), radius=0.15, color='yellow'))

                if (x-ps.extraArgs["goalX"])**2 + (y-ps.extraArgs["goalY"])**2 <= ps.extraArgs["goalRadius"]**2:
                    instance.drawnCircle.set(color='green', alpha=0.5)
                    ps.extraArgs["score"] += 1
                    ps.extraArgs["labelScore"].text = "Punts = {}".format(ps.extraArgs["score"])

                else:
                    instance.drawnCircle.set(color='red', alpha=0.5)
                    ps.extraArgs["lives"] -= 1
                    ps.extraArgs["labelLives"].text="Vides = {}".format(ps.extraArgs["lives"])#'♥' * ps.extraArgs["lives"]
                    if ps.extraArgs["lives"] == 0:
                        ps.stopPlaying()
                        ps.playButton.disabled_color = 'red'
                        ps.playButton.disabled = True
                        return instance.axPsi.text(0.5, 0.5, 'GAME OVER!', dict(size=30, fontweight=800, color='white'),
                                    horizontalalignment='center', verticalalignment='center',
                                    path_effects=[animate.peffects.withStroke(linewidth=4, foreground="black")],
                                    transform=instance.axPsi.transAxes)

                ps.stopPlaying()

                def newStep(*args):
                    ps.extraArgs["goalRadius"] = ps.extraArgs["goalRadius"] ** 0.9
                    ps.extraArgs["goalX"] = instance.QSystem.x0 + ps.extraArgs["goalRadius"] + random.random() * (
                                instance.QSystem.xf - instance.QSystem.x0 - 2 * ps.extraArgs["goalRadius"])
                    ps.extraArgs["goalY"] = instance.QSystem.y0 + ps.extraArgs["goalRadius"] + random.random() * (
                                instance.QSystem.yf - instance.QSystem.y0 - 2 * ps.extraArgs["goalRadius"])
                    instance.drawnCircle.remove()
                    ps.extraArgs["goalCircle"] = plt.Circle((ps.extraArgs["goalX"], ps.extraArgs["goalY"]),
                                                            ps.extraArgs["goalRadius"], alpha=0.2, color='black')
                    instance.drawnCircle = instance.axPsi.add_patch(ps.extraArgs["goalCircle"])
                    instance.observedParticle.remove()
                    #print(ps.extraArgs["score"])
                    ps.startPlaying()

                Clock.schedule_once(newStep, timeout=1)

                return instance.drawnCircle, instance.observedParticle

            """else:
                if instance.observedParticle is not None: instance.observedParticle.remove()"""

            return instance.drawnCircle

        """def extra_update_move(args):"""

        def extra_info_movement(args):
            layout = GridLayout(rows=4, width = 250, size_hint_x=None)

            args["labelScore"] = Label(text="Punts = {}".format(args["score"]))
            args["labelLives"] = Label(text="Vides = {}".format(args["lives"]))#'♥'*args["lives"])
            layout.add_widget(args["labelScore"])
            layout.add_widget(args["labelLives"])
            return layout




        definition = {
            #"QSystem": QSystem,
            "initState": inicial2D, "potential":potentialHarmonicWellMoving, "extra_param":extra_param,
            "drawClassical": False, "drawExpected": False, "duration": None,#10.
            #"extraCommands": [('key_press_event', moveHarmonicWellKeyboard)],
            "extraUpdates": [drawCircle], "isFocusable":False, "plotWidget":extra_info_movement,
            "showNorm": False, "showEnergy":False, "showMomentum":False, "debugTime":False, "extra_keyboard_action":extra_keyboard_movegame
        }
        """"initState": mathPhysics.eigenvectorHarmonic2DGenerator(0., 2, 0., 2, 8.),
            "potential": potentialHarmonicWellMoving
        }"""
        self.ids.gameSelect.add_widget(
            Button(text="Transporta la partícula!", on_release=partial(self.switch, definition=definition, setExtraArgs=setMoveGame)))


    def switch(self, *args, definition=None, setExtraArgs=None):
        self.manager.get_screen("playscreen").set_self(definition, setExtraArgs=setExtraArgs)
        self.manager.transition.direction = "left"
        self.manager.current = "playscreen"
        self.manager.get_screen("playscreen").sourceScreen = "games"




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
            scalePsi=anim.scalePsi, scaleMom=anim.scaleMom, isKivy=False,
            drawClassical=anim.drawClassical, drawClassicalTrace=anim.drawClassicalTrace, drawExpected=anim.drawExpected, drawExpectedTrace=anim.drawExpectedTrace)
        #animationToSave.reset_plot()
        try:    animationToSave.saveAnimation(fName, type)
        except: TextPopup("Error, format probablement no suportat").open()

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

    def setPotential(self):
        if (self.ids.potential.on_text_validate() == 0):
            self.window.QSystem.changePotential(self.window.potential)
            self.window.animation.manualUpdate(onlyDraw=True)
            #self.window.animation.reset_plot()

    def previewPotential(self):
        copy = self.window.potential
        copyDef = self.window.potentialDef

        if (self.ids.potential.on_text_validate() == 0):
            self.window.QSystem.setPotential(self.window.potential)

            self.window.potential = copy
            self.window.potentialDef = copyDef

            self.window.tempState["psi"] = self.window.QSystem.psiMod
            self.window.tempState["unit_col"] = unit_energy
            Factory.PlotPopup(self.window.tempState).open()

    def setInitState(self):
        if (self.ids.initState.on_text_validate() == 0):
            self.window.QSystem.setState(self.window.initState)
            self.window.animation.manualUpdate(onlyDraw=True)
            #self.window.animation.reset_plot()

    def previewInitState(self):
        copy = self.window.initState
        copyDef = self.window.initStateDef

        if (self.ids.initState.on_text_validate() == 0):
            self.window.QSystem.setTempState(self.window.initState)

            self.window.initState = copy
            self.window.initStateDef = copyDef

            self.window.tempState["psi"] = self.window.QSystem.psiCopy
            self.window.tempState["unit_col"] = r'${}^{{-2}}$'.format(unit_dist)
            Factory.PlotPopup(self.window.tempState).open()

    """def on_open(self):
        self.ids.Nx.text = str(self.window.animation.QSystem.Nx)
        self.ids.Ny.text = str(self.window.animation.QSystem.Ny)
        self.ids.x0.text = str(self.window.animation.QSystem.x0)
        self.ids.xf.text = str(self.window.animation.QSystem.xf)
        self.ids.y0.text = str(self.window.animation.QSystem.y0)
        self.ids.yf.text = str(self.window.animation.QSystem.yf)

        self.ids.t.text = str(self.window.animation.QSystem.t)

        self.ids.dtSim.text = str(self.window.animation.dtSim)"""

class PlayScreen(Screen):
    def __init__(self, sourceScreen = "examples", **kwargs):
        super(PlayScreen, self).__init__(**kwargs)
        self.sourceScreen = sourceScreen
        self.extra_param = np.array([1.])
        self.extra_on_enter = None
        self.animation = None

        # Keyboard: https://kivy.org/doc/stable/api-kivy.core.window.html
        self.extra_keyboard_action = None
        self._keyboard = None
        #self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        # self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None


    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if self.extra_keyboard_action is not None:
            self.extra_keyboard_action(self, keyboard, keycode, text, modifiers)

        return True

    def on_enter(self, *args):
        self.animation.reset_plot()
        if self.extra_on_enter is not None: self.extra_on_enter(self.extraArgs)

    def set_self(self, definition=None, setExtraArgs=None):
        #self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        #self._keyboard.bind(on_key_down=self._on_keyboard_down)

        if definition == None:
            definition = {}
        self.definition = definition

        self.extra_keyboard_action = None

        self.extra_on_enter = None

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
        self.variablesDict = {}
        self.setVarDict()

        self.initState = mathPhysics.gaussian2D(7, 1., self.extra_param[self.nVar - 2],
                                                7., 1., self.extra_param[self.nVar - 1])
        self.initStateDef = \
            "gaussianPacket(x, 7, 1, {px}) * gaussianPacket(y, 7, 1, {py})"
        # "1/(2*pi)**0.5 * exp(-1./4. * ((x-7)**2 + (y-7)**2)) * exp(1j * ({px}*x + {py}*y))"

        self.potential = mathPhysics.potentialBarrier
        self.potentialDef = "exp(-(x ** 2) / 0.1) * 5 / sqrt(0.1 * pi)"

        self.extra_update = None
        self.extra_clean = None

        for key in definition:
            vars(self)[key] = definition[key]

        if "QSystem" in self.definition:
            self.QSystem = self.definition["QSystem"]
        else:
            self.QSystem = mathPhysics.QuantumSystem2D(self.Nx, self.Ny, self.x0, self.y0, self.xf, self.yf,
                                                       self.initState,
                                                       potential=self.potential, extra_param=self.extra_param)

        animKeys = {"dtSim":0.01, "dtAnim":0.04, "debugTime":False, "duration":None,
                    "showPotential":True, "updatePotential":True, "showMomentum":True, "showEnergy":True, "showNorm":False,
                    "scalePsi":True, "scaleMom":True, "scalePot":True, "isFocusable":True,
                    "drawClassical":True, "drawClassicalTrace":False, "drawExpected":True, "drawExpectedTrace":False,
                    "extraCommands":[], "extraUpdates":[],
                    "unit_dist":unit_dist,"unit_mom":unit_mom,"unit_time":unit_time,"unit_energy":unit_energy}
        for key in animKeys:
            # Changes value to definition (dictionary), but if it's not in the dictionary leaves it as is (default)
            animKeys[key] = definition.get(key, animKeys[key])

        self.animation = animate.QuantumAnimation(self.QSystem, **animKeys,
                                                  isKivy=True)

        self.savedStates = []

        self.tempState = {"psi": self.QSystem.psi, "x0": self.QSystem.x0, "xf": self.QSystem.xf
            , "y0": self.QSystem.y0, "yf": self.QSystem.yf,
                          "name": "temp"}

        self.plotBox = BoxLayout(size_hint=(1, 0.8))
        self.plotBox.add_widget(self.animation.fig.canvas)

        self.setExtraArgs = setExtraArgs
        self.extraArgs = {}
        if setExtraArgs is not None:
            setExtraArgs(self.extraArgs)

        # ADD HERE EXTRA THINGS LIKE SLIDERS? CAN PASS DOWN CUSTOM WIDGET
        ######
        if "plotWidget" in definition:
            if callable(definition["plotWidget"]): self.plotBox.add_widget(definition["plotWidget"](self.extraArgs))   # Careful here with callable()
            else: self.plotBox.add_widget(definition["plotWidget"])

        ######


        buttonBox = BoxLayout(size_hint=(1, 0.2), orientation="horizontal", padding=10, spacing=20)

        resetButton = Button(text="Reset", on_release=self.resetAll)
        buttonBox.add_widget(resetButton)

        self.playButton = PlayButton(text="", state='normal' if self.paused else 'down',
                                  on_press= lambda x: self.startPlaying() if self.paused else self.stopPlaying())

        buttonBox.add_widget(self.playButton)

        returnButton = Button(text="Retorna enrere", on_press = self.goBack)
        buttonBox.add_widget(returnButton)

        mainBox = BoxLayout(orientation="vertical")

        mainBox.add_widget(self.plotBox)
        mainBox.add_widget(buttonBox)
        self.add_widget(mainBox)

    def goBack(self, *args):
        self.manager.transition.direction = "right"
        self.manager.current = self.sourceScreen

    def clean(self):
        self.stopPlaying()
        self.plotBox.clear_widgets()
        self.clear_widgets()
        plt.close(self.animation.fig)
        if self.extra_clean is not None: self.extra_clean(self.extraArgs)

    def on_leave(self):
        self.clean()

    def resetAll(self, *args):
        self.clean()
        self.set_self(self.definition, setExtraArgs=self.setExtraArgs)
        Clock.schedule_once(self.on_enter)

    def stopPlaying(self):
        try: self.schedule.cancel()
        except: pass
        self.paused = True
        self.animation.paused = True
        if self._keyboard is not None: self._keyboard.unbind(on_key_down=self._on_keyboard_down)

    def startPlaying(self):
        self.schedule = Clock.schedule_interval(self.play, self.animation.dtAnim)
        self.paused = False
        self.animation.paused = False
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        # self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def play(self, dt):
        if self.animation.paused:
            self.paused = self.animation.paused
            self.stopPlaying()
        else:
            self.animation.manualUpdate()
            if self.extra_update is not None: self.extra_update(self.extraArgs)

    def setVarDict(self):
        self.variablesDict.clear()
        for i in range(self.nVar):
            if self.paramNames[i] != "": self.variablesDict[self.paramNames[i]] = "extra_param[{}]".format(i)
        """self.variablesDict = \
            {self.paramNames[i]: "extra_param[{}]".format(i) for i in range(self.nVar) if self.paramNames[i] != ""}"""




class quantumMovementApp(App):
    def build(self):
        return WindowManager()
        #return kv


if __name__ == "__main__":
    plt.style.use("dark_background")
    quantumMovementApp().run()