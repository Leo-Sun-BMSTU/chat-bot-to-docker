from aiogram.dispatcher.filters.state import StatesGroup, State

class NewQuestion(StatesGroup):
    Question = State()
    Answer = State()

class NewLab(StatesGroup):
    Lab = State()