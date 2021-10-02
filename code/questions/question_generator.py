import numpy as np
import pandas as pd
import random

class Questions:
    """
    Класс генерирует случайные исходные данные для тестовых задач
    """
    datasets_task_1 = None
    datasets_task_2 = None
    parameters = {'task_1': None,
                  'task_2': None}
    def __init__(self):
        """
        Конструктор класса
        """
        self.parameters = None

    def generate_dataset(self):
        dataset = pd.DataFrame(data=np.random.randint(1, 10, (6, 3)),
                               columns=['x1', 'x2', 'x3'])
        dataset['y'] = np.random.randint(0, 2, (6, 1))
        last_row = pd.DataFrame(data=np.random.randint(1, 10, (1, 3)),
                                columns=['x1', 'x2', 'x3'])
        dataset = dataset.append(last_row,
                                 ignore_index=True)
        dataset = dataset.fillna('')
        return dataset

    def generate_neighbor_number(self):
        neighbor_number = random.randint(2, 6)
        return neighbor_number

    def get_task_1(self):
        dataset = self.generate_dataset()
        neighbor_number = self.generate_neighbor_number()
        Questions.datasets_task_1 = dataset
        Questions.parameters['task_1'] = neighbor_number
        text_of_task = f'    Провести классификацию неизвестного элемента по заданной выборке объектов \
методом {neighbor_number} ближайших соседей. Расстояние считать по Евклидовой метрике.'
        return text_of_task, dataset

    def get_task_2(self):
        dataset = self.generate_dataset()
        neighbor_number = self.generate_neighbor_number()
        Questions.datasets_task_2 = dataset
        Questions.parameters['task_2'] = neighbor_number
        text_of_task = f'    Провести классификацию неизвестного элемента по заданной выборке объектов \
методом {neighbor_number} ближайших соседей с функцией весов 𝑤(𝑥1, 𝑥2) = 1 / 𝑖, где i – номер соседа \
по близости к рассматриваемому элементу. в качестве расстояния использовать евклидову метрику.'
        return text_of_task, dataset

ques = Questions()
print(ques.get_task_1())