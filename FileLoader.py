import pandas as pd


class FileLoader(object):

    def __init__(self, filename):
        self.__filename = filename

    def set_filename(self, filename):
        self.__filename = filename

    def get_filename(self):
        return self.__filename

    def read_file(self):
        return pd.read_csv(self.__filename)



