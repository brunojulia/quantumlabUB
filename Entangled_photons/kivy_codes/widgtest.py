from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.factory import Factory

Builder.load_string('''
<BaseWidget>:
    orientation: 'vertical'
    widget_title: widget_title
    placeholder: placeholder
    Label:
        size_hint: None, 0.1
        id: widget_title
    Placeholder
        id: placeholder

<TwoButtonWidget>:
    orientation: 'vertical'
    button1: button1
    Button:
        text: 'button 1'
        id: button1
    Button:
        text: 'button 2'

<ThreeButtonWidget>:
    orientation: 'vertical'
    Button:
        text: 'button a'
    Button:
        text: 'button b'
    Button:
        text: 'button c'
''')

class BaseWidget(BoxLayout):
    def __init__(self, **args):
        # unregister if already registered...
        Factory.unregister('Placeholder')
        Factory.register('Placeholder', cls=self.placeholder)
        super(BaseWidget, self).__init__(**args)

class TwoButtonWidget(BoxLayout):
    pass

class ThreeButtonWidget(BoxLayout):
    pass

class CustomizedWidget1(BaseWidget):
    placeholder = TwoButtonWidget

class CustomizedWidget2(BaseWidget):
    placeholder = ThreeButtonWidget

class MyApp(App):
    def build(self):
        layout = BoxLayout()
        c1 = CustomizedWidget1()
        # we can access base widget...
        c1.widget_title.text = 'First'
        # we can access placeholder
        c1.placeholder.button1.text = 'This was 1 before'

        c2 = CustomizedWidget2()
        c2.widget_title.text = 'Second'

        layout.add_widget(c1)
        layout.add_widget(c2)
        return layout

if __name__ == '__main__':
    MyApp().run()