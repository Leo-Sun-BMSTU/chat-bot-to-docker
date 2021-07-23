import pandas as pd
import constants

class Prepropcess:
    def __init__(self,  data):
        """
        Конструктор класса.
        data - путь к файлу
        status - словарь, содержащий маркеры состояния данных:
                 True - данные удовлетворяют требованиям,
                 False - данные НЕ удовлетворяют требованиям.
        :param data:
        """
        self.data = data
        self.status = {
            'is_last_column_target': None,
            'row_number': None,
            'nan_duplicate_search': None,
            'normalization': None,
            'correlation': None,
            'data_type_checker': None
        }

    def is_last_column_target(self):
        """
        Метод создаёт dataframe и проверяет название последнего столбца
        :return:
        """
        read_data = pd.read_csv(self.data)
        last_column = read_data.columns[-1]
        self.status['is_last_column_target'] = last_column == constants.TARGET_VALUE_NAME
        return self.status['is_last_column_target']

    def row_number(self):
        '''
        Проверка число строк в dataframe. По условию их число должно быть в промежутко от 300 до 50000
        '''
        read_data = pd.read_csv(self.data)
        rows_number = read_data.shape[0]
        self.status['row_number'] = constants.MIN_ROW_NUMBER_THRESHOLD <= rows_number <= constants.MAX_ROW_NUMBER_THRESHOLD
        return self.status['row_number']

    def nan_duplicate_search(self):
        '''
        Проверка наличия пропусков и дубликато в dataframe
        '''
        read_data = pd.read_csv(self.data)
        nan_number = read_data.isnull().sum()
        duplicate_number = read_data.duplicated().sum()
        result = (nan_number == 0) & (duplicate_number == 0)
        self.status['nan_duplicate_search'] = all(result)
        return self.status['nan_duplicate_search']

    def normalization(self):
        '''
        Проверка на нормализованность данных, метод проверяте лежат ли данные
        в промежутке от -1 до 1
        '''
        read_data = pd.read_csv(self.data)
        result = (read_data > -1) & (read_data < 1)
        self.status['normalization'] = all(result)
        return self.status['normalization']

    def correlation(self):
        '''
        Проверка на наличие высококоррелирующих признаков
        '''
        read_data = pd.read_csv(self.data)
        corr_table = read_data.corr()
        result = (abs(corr_table) > constants.CORR_COEF) & (abs(corr_table) != 1)
        self.status['correlation'] = all(result)
        return self.status['correlation']

    def data_type_checker(self):
        '''
        Проверка отсутствия элементов типа object
        '''
        read_data = pd.read_csv(self.data)
        value_types = read_data.dtypes != object
        self.status['data_type_checker'] = all(value_types)
        return self.status['data_type_checker']

    def run(self):
        '''
        Последовательное исполнение методов класса
        '''
        self.is_last_column_target()
        self.row_number()
        self.nan_duplicat_search()
        self.normalization()
        self.correlation()
        self.data_type_checker()
        result = all(self.status.values())
        return result