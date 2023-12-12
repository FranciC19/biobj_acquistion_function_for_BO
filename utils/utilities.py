import os


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def pickle_folder_initialization(date: str, folder_name: str = None):
    make_folder(os.path.join('experiments', date))
    if folder_name is not None:
        make_folder(os.path.join('experiments', date, folder_name))
