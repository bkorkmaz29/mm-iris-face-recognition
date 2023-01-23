import PySimpleGUI as sg
import numpy as np
import os

from controllers import on_rec, on_add


def main():

    sg.theme('SandyBeach')

    main_layout = [
       [sg.Image(filename='cover.png', key='image')]
    ]


    rec_layout = [
        [sg.Text('Recognize', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Text('Face File'), sg.In(size=(25,1), enable_events=True ,key='-FACEFILE-'), sg.FileBrowse()],
        [sg.Text('Iris File'), sg.In(size=(25,1), enable_events=True ,key='-IRISFILE-'), sg.FileBrowse()],
        [sg.Button('Search')]
    ]


    add_layout = [
        [sg.Text('Add User', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Text('Name'), sg.InputText()],
        [sg.Text('Face File'), sg.In(size=(25,1), enable_events=True ,key='-FACEFILE2-'), sg.FileBrowse()],
        [sg.Text('Iris File'), sg.In(size=(25,1), enable_events=True ,key='-IRISFILE2-'), sg.FileBrowse()],
        [sg.Button('Add')]
    ]

    
    layout = [
        [sg.Button('Recognize'), sg.Button('Add User'), sg.Button('Exit')],
        [sg.Column(rec_layout, visible=False, key='rec'), sg.Column(add_layout, visible=False, key='add'), sg.Column(main_layout, visible=True, key='main')]
    ]

    window = sg.Window('app', layout, location=(300, 300), size=(900, 700))
 
    input_layout = [
        [sg.Text('Name', size =(15, 1)), sg.InputText()],
        [sg.Submit(), sg.Cancel()]
    ]
    

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return
        
        elif event == 'Recognize':
            window['main'].update(visible=False)
            window['rec'].update(visible=True) 
            window['add'].update(visible=False)        
     
        elif event == 'Search':
                on_rec(values["-FACEFILE-"], values["-IRISFILE-"]) 
        

                                     
        elif event == 'Add User':
            window['main'].update(visible=False)
            window['rec'].update(visible=False) 
            window['add'].update(visible=True)

        elif event == 'Add':
            on_add(values["-FACEFILE2-"], values["-IRISFILE2-"]) 
main()
