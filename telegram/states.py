from aiogram.dispatcher.filters.state import StatesGroup, State

class NewName(StatesGroup):
    Name = State()

class NewQuestion(StatesGroup):
    Question = State()
    Answer = State()

class NewLab(StatesGroup):
    Lab = State()