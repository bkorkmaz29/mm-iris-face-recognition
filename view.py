import PySimpleGUI as sg
import numpy as np
import os
dir_path = 'data/db'
folder_names = [name for name in os.listdir(
    dir_path) if os.path.isdir(os.path.join(dir_path, name))]


def refresh_folder_list(window):
    folder_names = [name for name in os.listdir(
        dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    window['-DB-'].update(values=folder_names)


def view(controller):

    sg.theme('BlueMono')

    rec_layout = [

        [
            sg.Frame('', [
                [
                    sg.Column([
                        [
                            sg.Frame('Input', layout=[
                                [
                                    sg.Text('Face Image', size=(9, 1), font=("Roboto", 10)),  sg.In(
                                        size=(25, 1), enable_events=True, key='-FACEREC-'), sg.FileBrowse()
                                ],
                                [
                                    sg.Text('Iris Image', size=(9, 1), font=("Roboto", 10)), sg.In(
                                        size=(25, 1), enable_events=True, key='-IRISREC-'), sg.FileBrowse()
                                ],
                                [
                                    sg.Button('Search', size=(10, 1),
                                              pad=((10, 0), (10, 0)))
                                ]
                            ], element_justification='center')
                        ]
                    ], element_justification='center', expand_y=True),

                    sg.VSeparator(),

                    sg.Frame('Settings', [
                        [
                            sg.Column([
                                [sg.Radio('Parallel', 'group1', key='-ORDER1-', enable_events=True,
                                          default=True, font=("Roboto", 10))],
                                [sg.Radio('Face-->Iris', 'group1', key='-ORDER2-', enable_events=True,
                                          font=("Roboto", 10))],
                                [sg.Radio('Iris-->Face', 'group1', key='-ORDER3-', enable_events=True,
                                          font=("Roboto", 10))]
                            ], element_justification='left', expand_y=True),
                            sg.VSeparator(),
                            sg.Column([
                                [sg.Radio('Precision', 'group2', key="-MODE1-", enable_events=True,
                                          font=("Roboto", 10))],
                                [sg.Radio('Balanced', 'group2', key="-MODE2-", enable_events=True,
                                          default=True, font=("Roboto", 10))],
                                [sg.Radio('Accuracy', 'group2', key="-MODE3-", enable_events=True,
                                          font=("Roboto", 10))]
                            ], element_justification='left', expand_y=True)
                        ]
                    ])
                ]
            ], element_justification='center', key='-FRAME', relief='raised')
        ],
        [
            sg.Frame('Results', [
                [sg.Text("", background_color='white', size=(24, 1), key='-TEXTSEARCHED-', font=("Roboto", 12), justification='center'),
                 sg.Image(filename='', background_color="white",
                          key='-FACEIMAGESEARCH-', size=(240, 160)),
                 sg.Image(filename='', background_color="white", key='-IRISIMAGESEARCH-', size=(240, 160))],
                [sg.Text("", background_color='white', size=(24, 1), key='-TEXTMATCHED-', font=("Roboto", 12), justification='center'),
                 sg.Image(filename='', background_color="white",
                          key='-FACEIMAGEMATCHED-', size=(240, 160)),
                 sg.Image(filename='', background_color="white", key='-IRISIMAGEMATCHED-', size=(240, 160))]
            ], border_width=3, size=(800, 600), background_color="white", key='-IMAGEFRAME',
                element_justification='center', vertical_alignment='center', relief="groove")
        ]
    ]

    enroll_layout = [
        [
            sg.Frame('', layout=[
                [
                     sg.Column([
                         [
                             sg.Frame('Subject Info', layout=[
                                 [
                                     sg.Frame('', layout=[
                                         [
                                             sg.Text('Name', size=(7, 1),
                                                     font=("Roboto", 10)),
                                             sg.Input(size=(25, 1),
                                                      key='-NAMEADD-')
                                         ]
                                     ], border_width=0, element_justification='center')
                                 ],
                                 [
                                     sg.Frame('', layout=[
                                         [
                                             sg.Text('Surname', size=(7, 1),
                                                     font=("Roboto", 10)),
                                             sg.Input(size=(25, 1),
                                                      key='-SURNAMEADD-')
                                         ]
                                     ], border_width=0, element_justification='center')
                                 ]
                             ], element_justification='center', expand_y=True),
                             sg.Frame('Input', layout=[
                                 [
                                     sg.Text('Face Image', size=(9, 1),
                                             font=("Roboto", 10)),
                                     sg.In(size=(25, 1),
                                           enable_events=True, key='-FACEADD-'),
                                     sg.FileBrowse()
                                 ],
                                 [
                                     sg.Text('Iris Image', size=(9, 1),
                                             font=("Roboto", 10)),
                                     sg.In(size=(25, 1),
                                           enable_events=True, key='-IRISADD-'),
                                     sg.FileBrowse()
                                 ],

                             ], element_justification='center')
                         ]
                     ], element_justification='center', expand_y=True),


                     ],
                [
                    sg.Button('Enroll', size=(10, 1),
                              pad=((0, 0), (7, 6)))
                ]
            ], border_width=1, relief='raised', element_justification='center', key='-FRAME-', vertical_alignment='center')
        ],

        [
            sg.Frame('Results', [
                [sg.Text('', font=('Roboto', 20), size=(
                     40, 1), justification='center', background_color="white", key='-TEXTADD-')],
                [sg.Image(filename='', key='-FACEIMAGEADD-', size=(320, 240), background_color='white'),
                 sg.Image(filename='', key='-IRISIMAGEADD-', size=(320, 240), background_color='white')],
            ], border_width=3, size=(800, 600), background_color="white", key='-IMAGEFRAME', element_justification='center', vertical_alignment='center', relief="groove")
        ]
    ]

    db_layout = [

        [sg.Frame('', [
            [sg.Frame('Select Subject', [
                [sg.Listbox(values=folder_names, size=(40, 4),
                            key='-DB-', enable_events=True)],
                [sg.Frame('', [[sg.Button('Display', pad=((0, 0), (0, 0))), sg.Button(
                    'Delete', pad=((10, 0), (0, 0)))]], element_justification='center', border_width=0, expand_x=True)],

            ], expand_x=True)]
        ], border_width=1, relief='raised', element_justification='center', key='-FRAME-')],

        [sg.Frame('Results', [
            [sg.Text('', font=('Roboto', 20), size=(
                40, 1), justification='center', background_color='white', key='-TEXTDB-')],
            [sg.Image(filename='', key='-FACEIMAGEDB-', size=(320, 240), background_color='white'),
             sg.Image(filename='', key='-IRISIMAGEDB-', size=(320, 240), background_color='white')],
        ], border_width=3, size=(800, 600), background_color='white', key='-IMAGEFRAME-', element_justification='center',  vertical_alignment='center', relief="groove")]
    ]
    tabgrp = [

        [
            [sg.TabGroup([
                [
                    sg.Frame('', layout=[
                        [sg.Button('Exit', button_color=('white', 'red'))]], border_width=0, element_justification='right', expand_x=True),
                ],
                [sg.Tab('Recognition', rec_layout, background_color='Ivory',
                        element_justification='center', key='-TAB-'),
                    sg.Tab('Enroll', enroll_layout, background_color='Ivory',
                           element_justification='center', key='-TAB-'),
                    sg.Tab("Database", db_layout, background_color='Ivory',
                           element_justification='center', key='-TAB-')
                 ]
            ], tab_location='centertop', border_width=5, tab_background_color='Gray', selected_background_color='Ivory', size=(1000, 500))]
        ],]

    window = sg.Window('app', tabgrp, element_justification='center',
                       location=(0, 0), size=(1200, 600))
    face_elem_add = window['-FACEIMAGEADD-']
    iris_elem_add = window['-IRISIMAGEADD-']
    face_elem_search = window['-FACEIMAGESEARCH-']
    iris_elem_search = window['-IRISIMAGESEARCH-']
    face_elem_matched = window['-FACEIMAGEMATCHED-']
    iris_elem_matched = window['-IRISIMAGEMATCHED-']
    face_elem_db = window['-FACEIMAGEDB-']
    iris_elem_db = window['-IRISIMAGEDB-']
    text_elem_add = window['-TEXTADD-']
    text_elem_db = window['-TEXTDB-']
    text_elem_search = window['-TEXTSEARCHED-']
    text_elem_matched = window['-TEXTMATCHED-']

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return
        elif event == 'Search':
            if values['-ORDER1-']:
                controller.set_order(1)
            elif values['-ORDER2-']:
                controller.set_order(2)
            elif values['-ORDER3-']:
                controller.set_order(3)

            if values['-MODE1-']:
                controller.set_mode(1)
            elif values['-MODE2-']:
                controller.set_mode(2)
            elif values['-MODE3-']:
                controller.set_mode(3)

            if not values['-FACEREC-'] or not values['-IRISREC-']:
                sg.popup('Please fill every field.', title='Error')
            else:
                face_img_search = controller.get_image(values["-FACEREC-"], 0)
                iris_img_search = controller.get_image(values["-IRISREC-"], 0)
                face_elem_search.update(data=face_img_search)
                iris_elem_search.update(data=iris_img_search)
                match_id, match_name, match_dir = controller.on_rec(
                    values["-FACEREC-"], values["-IRISREC-"])

                if match_id == 0:
                    text_elem_matched.update("No matching person")
                    face_elem_matched.update(data=None)
                    iris_elem_matched.update(data=None)
                else:
                    res = str(match_id)
                    face_img_matched, iris_img_matched = controller.get_db_image(
                        match_dir, 0)
                    face_elem_matched.update(data=face_img_matched)
                    iris_elem_matched.update(data=iris_img_matched)
                    text_elem_search.update("Searched")
                    text_elem_matched.update(match_name)
        elif event == 'Enroll':
            if not values['-FACEREC-'] or not values['-IRISREC-'] or not values['-NAMEADD-'] or not values['-SURNAMEADD-']:
                sg.popup('Please fill every field.', title='Error')
            else:
                full_name = values['-NAMEADD-'] + " " + values['-SURNAMEADD-']
                face_img = controller.get_image(values["-FACEADD-"], 1)
                iris_img = controller.get_image(values["-IRISADD-"], 1)
                face_elem_add.update(data=face_img)
                iris_elem_add.update(data=iris_img)
                id = controller.on_add(values["-FACEADD-"],
                                       values["-IRISADD-"], values['-NAMEADD-'], values['-SURNAMEADD-'])
                if id == 0:
                    text_elem_add.update("No face detected")
                else:
                    text_elem_add.update(
                        "Enrolled: " + full_name + " ID# " + id)
                refresh_folder_list(window)

        elif event == 'Display':
            user_info = controller.format_name(values["-DB-"][0])
            face_img, iris_img = controller.get_db_image(values["-DB-"][0], 1)
            face_elem_db.update(data=face_img)
            iris_elem_db.update(data=iris_img)
            text_elem_db.update(user_info)
        elif event == 'Delete':
            user_info = controller.format_name(values["-DB-"][0])
            result = controller.on_delete(values["-DB-"][0])
            face_elem_db.update(data=None)
            iris_elem_db.update(data=None)
            text_elem_db.update(user_info + " deleted")
            refresh_folder_list(window)
