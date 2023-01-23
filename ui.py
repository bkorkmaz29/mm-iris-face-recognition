import PySimpleGUI as sg
import numpy as np
import os


from controllers import on_rec, on_add, get_image
 
 
def main():
 
    sg.theme('BlueMono')
 
    rec_layout = [
        [sg.Text('Recognize', size=(40, 1),
                 justification='center', font='Helvetica 20')],
        [sg.Text('Face File', background_color='Ivory'), sg.In(
            size=(25, 1), enable_events=True, key='-FACEREC-'), sg.FileBrowse()],
        [sg.Text('Iris File', background_color='Ivory'), sg.In(
            size=(25, 1), enable_events=True, key='-IRISREC-'), sg.FileBrowse()],
        [sg.Button('Search')]
    ]
 
    add_layout = [
        [sg.Text('Add User', size=(40, 1),
                 justification='center', font='Helvetica 20')],
        [sg.Text('Name', background_color='Ivory'), sg.InputText(key='-NAME-')],
        [sg.Text('Face File', background_color='Ivory'), sg.In(
            size=(25, 1), enable_events=True, key='-FACEADD-'), sg.FileBrowse()],
        [sg.Text('Iris File', background_color='Ivory'), sg.In(
            size=(25, 1), enable_events=True, key='-IRISADD-'), sg.FileBrowse()],
        [sg.Button('Add')]
    ]
 
    tabgrp = [
        [sg.TabGroup([[
            sg.Button('Exit'),
            sg.Tab('Recognition', rec_layout, title_color='Red', border_width=10, background_color='Ivory', tooltip='Recognition',
                   element_justification='center'),
            sg.Tab('Add New User', add_layout, title_color='Blue', background_color='Ivory', element_justification='center')]], tab_location='centertop',
            border_width=5, tab_background_color='Gray', selected_background_color='Ivory')
 
         ],
        [sg.Frame('',[[
            sg.Text('No Results', font='Helvetica 20', key='-TEXT-')],
        [sg.Image(filename='', key='-FACEIMAGE-'),
         sg.Image(filename='', key='-IRISIMAGE-')]], key='-IMAGEFRAME', border_width=5)
    ]]
 
    window = sg.Window('app', tabgrp, element_justification='center',
                       location=(0, 0), size=(1200, 600))
    face_elem = window['-FACEIMAGE-']
    iris_elem = window['-IRISIMAGE-']
    text_elem = window['-TEXT-']
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return
 
        elif event == 'Search':
            face_img = get_image(values["-FACEREC-"])
            iris_img = get_image(values["-IRISREC-"])
            face_elem.update(data=face_img)
            iris_elem.update(data=iris_img)
            scores, result = on_rec(values["-FACEREC-"], values["-IRISREC-"])
            res = str(result)
            text_elem.update(res)
            
        elif event == 'Add':
            face_img = get_image(values["-FACEADD-"])
            iris_img = get_image(values["-IRISADD-"])
            face_elem.update(data=face_img)
            iris_elem.update(data=iris_img)
            text_elem.update(values['-NAME-'])
            on_add(values["-FACEADD-"], values["-IRISADD-"])
 
 
main()