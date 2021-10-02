import pickle
import pandas as pd

class Importer:
    data = property()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @data.deleter
    def data(self):
        del self._data

    def getFile(self, path):
        try:
            self._data = pd.read_csv(path, sep=',')
            print("File {} is dataframe".format(path))
            return self
        except pd.errors.ParserError:
            print("File {} is not dataframe".format(path))
        except FileNotFoundError:
            print("Incorrect file name")
        try:
            with open(path, 'rb') as f1:
                self._data = pickle.load(f1, encoding='utf-8')
                print("File {} is pickle".format(path))
                return self
        except ValueError:
            print("File {} is not pickle".format(path))
        except FileNotFoundError:
            print("Incorrect file name")
        except pickle.UnpicklingError:
            print("File {} is not pickle".format(path))