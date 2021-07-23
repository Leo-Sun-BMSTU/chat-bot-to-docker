from abc import ABCMeta, abstractmethod
from random import randint


class BaseQuestion(metaclass=ABCMeta):
    """
    Базовый класс для вопросов
    """
    __tasks = []

    @abstractmethod
    def get_task(self):
        raise NotImplementedError

    @abstractmethod
    def get_answer(self):
        raise NotImplementedError

    @classmethod
    def register_task(cls, task):
        cls._BaseQuestion__tasks.append(task)
        return task

    @classmethod
    def get_task_class(cls):
        index = randint(len(cls._BaseQuestion__tasks))
        return cls._BaseQuestion__tasks[index]

    @classmethod
    def get_random_task(cls):
        return cls.get_task_class()()