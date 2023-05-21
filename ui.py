import PySimpleGUI as sg
import numpy as np
import os


from controllers import on_rec, on_add, get_image, get_db_image, get_ns

dir_path = 'data/db'
folder_names = [name for name in os.listdir(
    dir_path) if os.path.isdir(os.path.join(dir_path, name))]


def refresh_folder_list(window):
    folder_names = [name for name in os.listdir(
        dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    window['-DB-'].update(values=folder_names)


def ui():

    sg.theme('BlueMono')

    rec_layout = [
        [sg.Text('Recognize', size=(80, 1),
                 justification='center', font='Helvetica 20')],
        [sg.Frame("", [[sg.Text("", size=(40, 1))],
                  [sg.Text('Face File'), sg.In(
                      size=(25, 1), enable_events=True, key='-FACEREC-'), sg.FileBrowse()],
                  [sg.Text('Iris File'), sg.In(
                      size=(25, 1), enable_events=True, key='-IRISREC-'), sg.FileBrowse()],
                  [sg.Button('Search')]], border_width=5, element_justification='center',  key='-FRAME')],
        [sg.Frame('', [

            [sg.Text("", background_color='Ivory', key='-TEXTSEARCHED-'), sg.Image(filename='', key='-FACEIMAGESEARCH-'),
             sg.Image(filename='', key='-IRISIMAGESEARCH-'), ],
            [sg.Text("", background_color='Ivory', key='-TEXTFOUND-'), sg.Image(filename='', key='-FACEIMAGEFOUND-'),
             sg.Image(filename='', key='-IRISIMAGEFOUND-')],
        ],

            border_width=0, element_justification='center', background_color="Ivory", key='-IMAGEFRAME')
        ]
    ]

    add_layout = [
        [sg.Text('Enroll', size=(80, 1),
                 justification='center', font='Helvetica 20')],
        [sg.Frame("", [[sg.Text("", size=(40, 1))],
                       [sg.Text('Name:'),
                       sg.Input(key='-NAMEADD-')],
                       [sg.Text('Face File'), sg.In(
                           size=(25, 1), enable_events=True, key='-FACEADD-'), sg.FileBrowse()],
                       [sg.Text('Iris File'), sg.In(
                           size=(25, 1), enable_events=True, key='-IRISADD-'), sg.FileBrowse()],
                       [sg.Button('Enroll')]], border_width=5, element_justification='center',  key='-FRAME')],
        [sg.Text("", size=(40, 2), background_color='Ivory')],
        [sg.Frame('', [
            [sg.Text('', font='Helvetica 20', size=(40, 1),
                     justification='center', background_color="Ivory", key='-TEXTADD-')],
            [sg.Image(filename='', key='-FACEIMAGEADD-'),
             sg.Image(filename='', key='-IRISIMAGEADD-')],
        ],
            border_width=0, background_color="Ivory", key='-IMAGEFRAME')
        ]
    ]

    db_layout = [
        [sg.Text('Database', size=(80, 1),
                 justification='center', font='Helvetica 20')],
        [sg.Frame("", [[sg.Text("", size=(40, 1))],
                       [sg.Listbox(values=folder_names,
                                   size=(20, 5), key='-DB-')],
                       [sg.Button('Display'), sg.Button('Delete')]], border_width=5, element_justification='center',  key='-FRAME')],

        [sg.Frame('', [
            [sg.Text('', font='Helvetica 20', size=(40, 1),
                     justification='center', background_color="Ivory", key='-TEXTDB-')],
            [sg.Image(filename='', key='-FACEIMAGEDB-'),
             sg.Image(filename='', key='-IRISIMAGEDB-')],
        ],
            border_width=0, background_color="Ivory", key='-IMAGEFRAME')
        ]

    ]

    tabgrp = [
        [sg.TabGroup([[
            sg.Button('Exit'),
            sg.Tab('Recognition', rec_layout, background_color='Ivory',
                   element_justification='center', key='-TAB-'),
            sg.Tab('Add New User', add_layout,
                   background_color='Ivory', element_justification='center', key='-TAB-'),
            sg.Tab("Database", db_layout,
                   background_color='Ivory', element_justification='center', key='-TAB-')
        ]], tab_location='centertop', border_width=5, tab_background_color='Gray', selected_background_color='Ivory', size=(1000, 500)),],
    ]

    window = sg.Window('app', tabgrp, element_justification='center',
                       location=(0, 0), size=(1200, 600))
    face_elem_add = window['-FACEIMAGEADD-']
    iris_elem_add = window['-IRISIMAGEADD-']
    face_elem_search = window['-FACEIMAGESEARCH-']
    iris_elem_search = window['-IRISIMAGESEARCH-']
    face_elem_found = window['-FACEIMAGEFOUND-']
    iris_elem_found = window['-IRISIMAGEFOUND-']
    face_elem_db = window['-FACEIMAGEDB-']
    iris_elem_db = window['-IRISIMAGEDB-']
    text_elem_add = window['-TEXTADD-']
    text_elem_db = window['-TEXTDB-']
    text_elem_search = window['-TEXTSEARCHED-']
    text_elem_found = window['-TEXTFOUND-']
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Search':
            face_img_search = get_image(values["-FACEREC-"], 0)
            iris_img_search = get_image(values["-IRISREC-"], 0)
            face_elem_search.update(data=face_img_search)
            iris_elem_search.update(data=iris_img_search)
            scores, result = on_rec(values["-FACEREC-"], values["-IRISREC-"])
            print(result)
            if result == 0:
                text_elem_add.update("No matching person")
                face_elem_found.update(data=None)
                iris_elem_found.update(data=None)
                text_elem_found.update("")
            else:
                res = str(result)
                face_img_found, iris_img_found = get_db_image(res, 0)
                face_elem_found.update(data=face_img_found)
                iris_elem_found.update(data=iris_img_found)
                text_elem_add.update("id: " + res)
                text_elem_search.update("Searched:")
                text_elem_found.update("Found:   ")

        elif event == 'Enroll':
            face_img = get_image(values["-FACEADD-"], 1)
            iris_img = get_image(values["-IRISADD-"], 1)
            face_elem_add.update(data=face_img)
            iris_elem_add.update(data=iris_img)
            id = on_add(values["-FACEADD-"],
                        values["-IRISADD-"], values["-NAMEADD-"])
            if id == 0:
                text_elem_add.update("No face detected")
            else:
                text_elem_add.update("Added as " + id)
            refresh_folder_list(window)

        elif event == 'Display':
            face_img, iris_img = get_db_image(values["-DB-"][0], 1)
            face_elem_db.update(data=face_img)
            iris_elem_db.update(data=iris_img)
            text_elem_db.update("id: " + values["-DB-"][0])


ui()
