from controller import Controller
from models.mm import Multimodal
from view import view

if __name__ == '__main__':
    multimodal = Multimodal()
    controller = Controller(multimodal)
    view(controller)
