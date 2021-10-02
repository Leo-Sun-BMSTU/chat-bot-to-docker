import random
from code.questions.base_question import BaseQuestion
from numpy.random import randint
from numpy.random import uniform
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import MaxPooling2D
from sklearn.tree import DecisionTreeRegressor

@BaseQuestion.register_task
class Task1(BaseQuestion):

    def __init__(self):
        """
        Конструктор класса.
        dataset - набор данных для анализа в рамках задания
        neighbor_number - гиперпараметр число соседей для модели k-ближайших соседей
        answer - верный ответ
        """
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (6, 3)), columns=columns)
        y_train = pd.Series(randint(0, 2, 6))
        X_test = pd.DataFrame(data=randint(1, 10, (1, 3)), columns=columns)
        dataset = X_train
        dataset = dataset.append(X_test, ignore_index=True)
        dataset['y'] = y_train
        dataset = dataset.fillna('')
        self.dataset = dataset
        self.neighbor_number = random.choice([3, 5])
        clf = KNeighborsClassifier(n_neighbors=self.neighbor_number)
        clf.fit(X_train, y_train)
        self.answer = clf.predict(X_test)[0]

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Провести классификацию неизвестного элемента по заданной выборке объектов \
методом {self.neighbor_number} ближайших соседей. Расстояние считать по Евклидовой метрике. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        """
        Метод возращает верный ответ.
        :return: str
        """
        return str(self.answer)

@BaseQuestion.register_task
class Task2(BaseQuestion):

    def __init__(self):
        """
        Конструктор класса.
        dataset - набор данных для анализа в рамках задания
        neighbor_number - гиперпараметр число соседей для модели k-ближайших соседей
        answer - верный ответ
        """
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (6, 3)), columns=columns)
        y_train = pd.Series(randint(0, 2, 6))
        X_test = pd.DataFrame(data=randint(1, 10, (1, 3)), columns=columns)
        dataset = X_train
        dataset = dataset.append(X_test, ignore_index=True)
        dataset['y'] = y_train
        dataset = dataset.fillna('')
        self.dataset = dataset

        self.neighbor_number = random.choice([3, 5])
        clf = KNeighborsClassifier(n_neighbors=self.neighbor_number, weights='distance')
        clf.fit(X_train, y_train)
        self.answer = clf.predict(X_test)[0]

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Провести классификацию неизвестного элемента по заданной выборке объектов \n\
методом {self.neighbor_number} ближайших соседей с функцией весов 𝑤(𝑥1, 𝑥2) = 1 / 𝑖, где i – номер соседа \n\
по близости к рассматриваемому элементу. В качестве расстояния использовать евклидову метрику. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task3(BaseQuestion):

    def __init__(self):
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (6, 3)), columns=columns)
        y_train = pd.Series(randint(10, 20, 6))
        X_test = pd.DataFrame(data=randint(1, 10, (1, 3)), columns=columns)
        dataset = X_train
        dataset = dataset.append(X_test, ignore_index=True)
        dataset['y'] = y_train
        dataset = dataset.fillna('')
        self.dataset = dataset

        self.neighbor_number = random.choice([2, 3, 4, 5])
        reg = KNeighborsRegressor(n_neighbors=self.neighbor_number, metric='minkowski', p=1)
        reg.fit(X_train, y_train)

        self.answer = reg.predict(X_test)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Провести регрессию неизвестного элемента по заданной выборке объектов \
методом {self.neighbor_number} ближайших соседей по метрике Минковского 1-й степени. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer[0])

@BaseQuestion.register_task
class Task4(BaseQuestion):

    def __init__(self):
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (10, 3)), columns=columns)
        y_train = pd.Series(randint(0, 2, 10))
        X_test = pd.DataFrame(data=randint(1, 10, (2, 3)), columns=columns)
        y_test = pd.Series(randint(0, 2, 2))

        train_dataset = X_train
        train_dataset['y'] = y_train
        self.train_dataset = train_dataset

        test_dataset = X_test
        test_dataset['y'] = y_test
        self.test_dataset = test_dataset

        clf = KNeighborsClassifier()
        parameters = {'n_neighbors': list(range(1, 6))}
        best_clf = GridSearchCV(clf, parameters, cv=4)
        best_clf.fit(X_train, y_train)
        self.answer = best_clf.best_params_['n_neighbors']


    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Подобрать оптимальный гиперпараметр kNN-классификатора по доле верных \
ответов, если известны обучающая и тестовая выборки. \n Обучающая выборка: \n {self.train_dataset.to_string()} \n \
Тестовая выборка: \n {self.test_dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task5(BaseQuestion):

    def __init__(self):
        predictions = pd.Series(randint(0, 2, 16))
        true_classes = pd.Series(randint(0, 2, 16))
        self.dataset = pd.DataFrame([predictions, true_classes],
                                    index=['Прогноз a(x)', 'Истинный класс y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: '', 7: '',
                                                    8: '', 9: '', 10: '', 11: '', 12: '', 13: '', 14: '', 15: ''})

        precision = precision_score(true_classes, predictions)
        recall = recall_score(true_classes, predictions)
        accuracy = accuracy_score(true_classes, predictions)
        f_score = f1_score(true_classes, predictions)

        self.answer = list(np.round(np.array([precision, recall, accuracy, f_score]), 2))
        self.answer = list(map(str, self.answer))

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Посчитать метрики precision, recall, accuracy, f-score для следующих ответов \n\
классификатора относительно класса 1. Ответ запишите с точностью до 2 знаков после запятой через пробел. \n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return self.answer[0], self.answer[1], self.answer[2], self.answer[3],

@BaseQuestion.register_task
class Task6(BaseQuestion):

    def __init__(self):
        predictions = pd.Series(uniform(0, 1, 7)).round(2)
        true_classes = pd.Series(randint(0, 2, 7))
        self.dataset = pd.DataFrame([predictions, true_classes],
                                    index=['Вероятность принадлежности объекта к классу 1, p',
                                                                        'Истинный класс, y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: ''})

        roc_auc = roc_auc_score(true_classes, predictions)

        # не понимаю как это работатет, но точно выдаёт площадь под ROC-кривой
        self.answer = '%0.2f' % roc_auc

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Рассчитать площадь под ROC-кривой, с точностью до сотых, для следующих предсказаний классификатора: \n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return self.answer

@BaseQuestion.register_task
class Task7(BaseQuestion):

    def __init__(self):
        predictions = pd.Series(uniform(0, 1, 7)).round(2)
        true_classes = pd.Series(randint(0, 2, 7))
        self.dataset = pd.DataFrame([predictions, true_classes],
                                    index=['Вероятность принадлежности объекта к классу 1, p', 'Истинный класс, y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: ''})

        precision, recall, thresholds = precision_recall_curve(true_classes, predictions, pos_label=1)
        index_threshold = np.where(precision[:-1].max())[0][0]

        self.answer = thresholds[index_threshold]

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Выбрать порог отнесения объекта к классу 1 для максимизации точности (precision) \n\
для следующих предсказаний. Ответ записать с точностью до двух знаков после запятой. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task8(BaseQuestion):

    def __init__(self):
        predictions = pd.Series(uniform(0, 1, 7)).round(2)
        true_classes = pd.Series(randint(0, 2, 7)).round(0)
        self.dataset = pd.DataFrame([predictions, true_classes],
                                    index=['Вероятность принадлежности объекта к классу 1, p',
                                           'Истинный класс, y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '', 6: ''})

        precision, recall, thresholds = precision_recall_curve(true_classes, predictions, pos_label=1)
        index_threshold = np.where(recall[:-1].max())[0][0]

        self.answer = thresholds[index_threshold]

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Выбрать порог отнесения объекта к классу 1 для максимизации  полноты (recall) \n\
для следующих предсказаний. Ответ записать с точностью до двух знаков после запятой. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task9(BaseQuestion):

    def __init__(self):
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (5, 3)), columns=columns)
        y_train = pd.Series(randint(0, 10, 5))
        self.dataset = pd.concat([X_train, y_train], axis=1)
        self.dataset = self.dataset.rename(columns={'x1': 'x1', 'x2': 'x2', 'x3': 'x3', 0: 'y'})

        reg = LinearRegression()
        reg.fit(X_train[['x1', 'x2', 'x3']], y_train)

        coefficients = reg.coef_
        self.answer = np.round(coefficients, 2).tolist()

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Найти оптимальные веса линейной регрессии, построенной по следующей \n\
тренировочной выборке. Ответ запишите с точностью до двух знаков после запятой через пробел. \n {self.dataset}'
        return text_of_task

    def get_answer(self):
        return list(map(str, self.answer))

@BaseQuestion.register_task
class Task10(BaseQuestion):

    def __init__(self):
        columns = ['x1', 'x2', 'x3']
        X_train = pd.DataFrame(data=randint(1, 10, (3, 3)), columns=columns)
        y_train = pd.Series(randint(1, 10, 3))
        self.dataset = pd.concat([X_train, y_train], axis=1)
        self.dataset = self.dataset.rename(columns={'x1': 'x1', 'x2': 'x2', 'x3': 'x3', 0: 'y'})

        zero_weights = np.zeros((4, 1))
        self.step = uniform(0.1, 1, 1).round(1)[0]
        self.answer = zero_weights - 2/3 * self.step * np.array([[sum(self.dataset.y)],
                                                            [sum(self.dataset.y * self.dataset.x1)],
                                                            [sum(self.dataset.y * self.dataset.x2)],
                                                            [sum(self.dataset.y * self.dataset.x3)]])
        self.answer = np.round(self.answer, 1)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Для линейной регрессии найти веса следующего шага градиентного спуска на \n\
данном наборе данных, если веса на текущем шаге равны 0. Функционал ошибки – \n\
среднеквадратичная ошибка. Шаг градиентного спуска принять равным {self.step}. \n\
Ответ запишите с точностью до одного знака после запятой через пробел \n {self.dataset}'
        return text_of_task

    def get_answer(self):
        return list(map(str, self.answer[:, 0].tolist()))

@BaseQuestion.register_task
class Task11(BaseQuestion):

    def __init__(self):
        self.weights = uniform(0, 1, (1, 4)).round(1)
        self.features = np.concatenate([np.ones(1), randint(-20, 20, 3)])

        probabilistic_predictions = np.exp(np.dot(self.weights, self.features)) / (1 + np.exp(np.dot(self.weights, self.features)))
        self.answer = probabilistic_predictions[0].round(2)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Найти вероятность принадлежности объекта классу 1, если классификация \
проводится логистической регрессией с вектором весов \n [w0, w1, w2, w3] = {np.array2string(self.weights[0])}. \
Объект задан признаками [x1, x2, x3] = {np.array2string(self.features[1:])}.'
        return text_of_task

    def get_answer(self):
        return np.array2string(self.answer)[0]

from numpy.random import choice

@BaseQuestion.register_task
class Task12(BaseQuestion):

    def __init__(self):
        predictions = randint(-10, 30, 3)
        true_class = choice([-1, 1], size=3)
        self.dataset = pd.DataFrame([predictions, true_class],
                                    index=['Прогноз модели', 'Истинный класс, y'],
                                    columns=['0', '1', '2'])
        self.dataset = self.dataset.rename(columns={'0': '', '1': '', '2': ''})

        self.answer = np.log(np.exp(-self.dataset.iloc[-1, :] * self.dataset.iloc[0, :]) + 1).sum() / self.dataset.shape[1]
        self.answer = round(self.answer, 2)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Посчитать логистическую функцию потерь для известных результатов линейной регрессии. \n\
Ответ запишите с точностью до двух знаков после запятой. \n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

from sklearn.tree import DecisionTreeClassifier

@BaseQuestion.register_task
class Task13(BaseQuestion):

    def __init__(self):
        X_train = pd.Series(list(range(1, 13)))
        y_train = pd.Series(randint(0, 2, 12))
        self.dataset = pd.DataFrame([X_train, y_train],
                                    index=['x', 'y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '',
                                            6: '', 7: '', 8: '', 9: '', 10: '', 11: ''})

        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X_train.values.reshape(-1, 1), y_train)
        self.answer = clf.get_depth()

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Какова будет максимальная глубина двоичного решающего дерева, построенного \
по энтропийному критерию ошибки на следующем наборе данных. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task14(BaseQuestion):

    def __init__(self):
        X_train = pd.Series(list(range(1, 13)))
        y_train = pd.Series(randint(0, 2, 12))
        self.dataset = pd.DataFrame([X_train, y_train], index=['x', 'y'], columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '',
                                                    6: '', 7: '', 8: '', 9: '', 10: '', 11: ''})

        self.min_samples_leaf = randint(1, 4)

        clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=self.min_samples_leaf)
        clf.fit(X_train.values.reshape(-1, 1), y_train)
        self.answer = round(clf.score(X_train.values.reshape(-1, 1), y_train), 2)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Построить двоичное решающее дерево по энтропийному критерию ошибки с \n\
минимальным количеством элементов в листе равном {self.min_samples_leaf}, на следующем наборе данных. \n\
Какая точность классификации будет у этого дерева на обучающей выборке? Ответ дайте с точностью до сотых.\n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task15(BaseQuestion):

    def __init__(self):
        X_train = pd.Series(randint(0, 20, 12))
        y_train = pd.Series(randint(0, 10, 12))
        X_test = pd.Series(randint(0, 25))
        self.dataset = pd.DataFrame([X_train.append(X_test, ignore_index=True), y_train],
                                    index=['x', 'y'],
                                    columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).fillna('')
        self.dataset = self.dataset.rename(columns={0: '', 1: '', 2: '', 3: '', 4: '', 5: '',
                                                    6: '', 7: '', 8: '', 9: '', 10: '', 11: '', 12: ''})

        self.depth = randint(2, 4)

        reg = DecisionTreeRegressor(max_depth=self.depth)
        reg.fit(X_train.values.reshape(-1, 1), y_train)
        self.answer = round(reg.predict(X_test.values.reshape(-1, 1))[0], 0)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Построить решающее дерево для задачи регрессии до глубины {self.depth} по следующему \n\
набору данных и провести регрессию неизвестного элемента. \n {self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return np.array2string(self.answer)[0]

@BaseQuestion.register_task
class Task16(BaseQuestion):

    def __init__(self):
        X_train = pd.Series(list(range(1, 13)), index=list(range(1, 13)))
        y_train = pd.Series(randint(0, 2, 12), index=list(range(1, 13)))
        self.X_test = pd.Series(np.round(uniform(1, 10), 2))

        self.dataset = pd.DataFrame([X_train, y_train], index=['x', 'y'], columns=list(range(1, 13)))

        self.datasubset_1 = self.dataset[X_train.sample(n=6, replace=False).values]
        self.datasubset_2 = self.dataset[X_train.sample(n=6, replace=False).values]
        self.datasubset_3 = self.dataset[X_train.sample(n=6, replace=False).values]

        clf = DecisionTreeClassifier()
        answer1 = clf.fit(self.datasubset_1.iloc[0, :].values.reshape(-1, 1),
                          self.datasubset_1.iloc[1, :].values.reshape(-1, 1)).predict(self.X_test.values.reshape(-1, 1))
        answer2 = clf.fit(self.datasubset_2.iloc[0, :].values.reshape(-1, 1),
                          self.datasubset_2.iloc[1, :].values.reshape(-1, 1)).predict(self.X_test.values.reshape(-1, 1))
        answer3 = clf.fit(self.datasubset_3.iloc[0, :].values.reshape(-1, 1),
                          self.datasubset_3.iloc[1, :].values.reshape(-1, 1)).predict(self.X_test.values.reshape(-1, 1))
        self.answer = round(np.array([answer1, answer2, answer3]).mean(), 0).astype(int)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Сделать прогноз для объекта x = {str(self.X_test[0])} путем случайного леса, состоящего из \n\
трех решающих деревьев, построенных по подвыборкам: \n\
x1 = {self.datasubset_1.iloc[0, :].tolist()}, \n\
х2 = {self.datasubset_2.iloc[0, :].tolist()} \n\
x3 = {self.datasubset_3.iloc[0, :].tolist()} \nдля следующего набора данных: \n\
{self.dataset.rename(columns={1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: "", 11: "", 12: ""}).to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task17(BaseQuestion):

    def __init__(self):
        self.subdatasets = pd.DataFrame(np.round(uniform(8, 10, (3, 4)), 2),
                                        columns=['', '', '', ''],
                                        index=['Подвыборка_1', 'Подвыборка_2', 'Подвыборка_3'])
        self.answer_mean = self.subdatasets.mean(axis=1).mean()
        self.answer_variance = self.subdatasets.mean(axis=1).var() * self.subdatasets.shape[1]

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Из большого набора данных (генеральной совокупности) взяли несколько подвыборок. \n\
В каждую из подвыборок попало несколько значений целевой переменной для одного и того же набора признаков. \n\
Оценить математическое ожидание и дисперсию целевой переменной в генеральной совокупности при этом наборе признаков. \n\
Ответ запишите с точностью до двух знаков после запятой через пробел. Пример: 5.5 3.8. \n\
{self.subdatasets.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(round(self.answer_mean, 2)), str(round(self.answer_variance, 2))

@BaseQuestion.register_task
class Task18(BaseQuestion):

    def __init__(self):
        self.dataset = pd.DataFrame(np.round(uniform(8, 10, (3, 4)), 2),
                                    columns=['', '', '', ''])
        self.true_value = round(uniform(8, 10), 2)
        mean = self.dataset.values.mean()
        self.answer_variance = round(self.dataset.values.var(), 2)
        self.answer_bias = round(abs(mean - self.true_value), 2)

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Регрессор построил следующие предсказания. Пусть истинное значение искомой \n\
переменной равно {self.true_value}. Найти составляющие ошибки предсказания: bias и variance. \n\
Ответ запишите с точностью до двух знаков после запятой через пробел. Пример: 5.5 3.8. \n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer_bias), str(self.answer_variance)

@BaseQuestion.register_task
class Task19(BaseQuestion):

    def __init__(self):
        self.target_value = pd.DataFrame([randint(4, 10, 5)], columns=['', '', '', '', ''], index=['y'])
        self.start_value = randint(1, 6)
        mean_shift = (self.target_value - self.start_value).mean(axis=1)

        self.answer = self.start_value + mean_shift

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Решаем задачу регрессии с помощью некоторого константного алгоритма b(x) = w. \n\
У этого алгоритма единственный параметр – значение константы w. Начальное \n\
приближение w = {self.start_value}. Провести итерацию градиентного бустинга над этим \n\
алгоритмом и записать новое значение w, если целевая переменная принимает значения: \n\
{self.target_value.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer[0])

@BaseQuestion.register_task
class Task20(BaseQuestion):

    def __init__(self):
        self.dataset = pd.DataFrame([np.round(uniform(-1, 1, 3), 1),
                                np.round(uniform(-1, 1, 3), 1),
                                randint(1, 6, 2)],
                                columns=['1', '2', '3'],
                                index=['w1', 'w2', 'x']).fillna('')
        y_s = 1 / (1 + np.exp(np.float64(np.array(self.dataset.iloc[-1, :-1].sum() * -self.dataset.iloc[0]))))
        self.answer = 1 / (1 + np.exp(-sum(y_s * self.dataset.iloc[1, :])))

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Построить предсказание двухслойной полносвязной нейросети прямого \n\
распространения с функцией активации «сигмоид» для следующих параметров сети и значений признаков объектов.\n\
Ответ запишите с точностью до двух знаков после запятой \n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(round(self.answer, 2))

@BaseQuestion.register_task
class Task21(BaseQuestion):

    def __init__(self):
        arr = randint(0, 256, (9, 9))
        self.image = pd.DataFrame(arr,
                                  columns=['', '', '', '', '', '', '', '', ''],
                                  index=['', '', '', '', '', '', '', '', ''])
        arr = arr.reshape(1, 9, 9, 1)
        max_pool = MaxPooling2D(pool_size=3, strides=2)
        model = Sequential([max_pool])
        pooled_image = pd.DataFrame(model.predict(arr).reshape(4, 4))
        self.answer = np.diag(pooled_image).sum()

    def get_task(self):
        """
        Метод генерирует текстовое задание.
        :return: str
        """
        text_of_task = f'Провести свертку следующего изображения макспуллингом сеткой размера 3х3 пикселя с шагом 2. \n\
В качестве ответа ввести сумму элементов диагонали получившегося массива. \n\
{self.image.to_string()}'
        return text_of_task

    def get_answer(self):
        return str(self.answer)

@BaseQuestion.register_task
class Task22(BaseQuestion):

    def __init__(self):
        self.dataset = pd.DataFrame(randint(1, 10, (5, 3)), columns=['x1', 'x2', 'x3'])
        self.dataset['y'] = randint(0, 2, 5)

        clf = MLPClassifier(hidden_layer_sizes=1, activation='relu')
        clf.fit(self.dataset[['x1', 'x2', 'x3']], self.dataset.y)
        self.answer = list(map(str, np.round(np.concatenate((clf.coefs_[0], clf.coefs_[1])).reshape(1, 4)[0], 2).tolist()))

    def get_task(self):
        text_of_task = f'Подобрать оптимальные веса перцептрона для классификации следующего набора данных. \n\
Функция активации – ReLu. Ответы запишите через пробел с точностью до двух знаков после запятой.\n\
{self.dataset.to_string()}'
        return text_of_task

    def get_answer(self):
        return self.answer