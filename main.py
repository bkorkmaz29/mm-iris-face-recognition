from controller import Controller
from models.mm import MMR
from view import view
DATA_DIR = "db/"
if __name__ == '__main__':
    multimodal = MMR(DATA_DIR)
    controller = Controller(multimodal)
    view(controller)
