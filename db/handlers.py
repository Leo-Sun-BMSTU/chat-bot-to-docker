import asyncio
import datetime
import random
import yaml
from datetime import datetime

from asyncio import sleep
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import CommandStart
from aiogram.types import Message

import database
from config import admin_id
from load_all import dp, bot

from states import NewQuestion, NewLab
from database import Lab, User, Test


db = database.DBCommands()


@dp.message_handler(CommandStart())
async def register_user(message: types.Message):
    chat_id = message.from_user.id
    group_id = message.get_args()
    user = await db.add_new_user(group_id=group_id)
    id = user.id
    count_users = await db.count_users()

    bot_username = (await bot.me).username
    bot_link = f"https://t.me/{bot_username}?start={id}"

    if group_id:
        text = ("Приветствую Вас!\n"
                "Количество пользователей в базе данных: {count_users}\n"
                "\n"
                "Пройти тестирование: /test").format(
            count_users=count_users,
        )
    else:
        text = ("Приветствую Вас!\n"
             "Количество пользователей в базе данных: {count_users}\n"
             "\n"
             "Ссылка для участников Вашей группы: {bot_link}\n"
             "Посмотреть участников Вашей группы: /members\n"
             "Проверить лабораторную работу: /labs\n"
             "Пройти тестирование: /test").format(
        count_users=count_users,
        bot_link=bot_link
        )
    if message.from_user.id == admin_id:
        text += ("\n"
                  "\nОпция для администратора: /somecommand")
    await bot.send_message(chat_id, text)


@dp.message_handler(commands=["cancel"], state=NewQuestion)
async def cancel(message: types.Message, state: FSMContext):
    await message.answer("Вы отменили тестирование. Для повторной попытки введите /test.")
    await state.reset_state()

@dp.message_handler(commands=["cancel"], state=NewLab)
async def cancel(message: types.Message, state: FSMContext):
    await message.answer("Вы отменили проверку. Для повторной попытки введите /labs.")
    await state.reset_state()

@dp.message_handler(commands=["test"])
async def test(message: types.Message):
    await message.answer("Для прожолжения нажмите /enter или введите любой текст. Для отмены нажмите /cancel.")
    await NewQuestion.Question.set()

@dp.message_handler(state=NewQuestion.Question)
async def question(message: types.Message, state: FSMContext):
    user = types.User.get_current()
    n = 6  # Кол-во всех вопросов для тестирования
    m = 5  # Кол-во вопросов, на которые должен ответить пользователь

    questions_given = await db.check_questions()
    if questions_given == m:
        text = "Тестирование пройдено!"
        await message.answer(text)
        await state.reset_state()
    else:
        var = await db.row()
        while True:
            k = 0
            r_r = random.choice(range(n))
            for i in var:
                if int(i[0])-1 == r_r:
                    k = 1
                    break
            if k == 1:
                continue
            else:
                break
        item = Test()
        with open("answers.yml", 'r', encoding='utf-8') as answers:
            a = yaml.full_load(answers)
            question_number = f"Вопрос номер {r_r + 1}"
            await message.answer(question_number)
            for i, j in a.items():
                answer = ""
                answer += f"{j[r_r]} \n"
            dt = datetime.now()
            item.question = r_r+1
            item.right_answer = answer
            item.date_time = dt
            item.user_id = user.id
        with open("questions.yml", 'r', encoding='utf-8') as questions:
            q = yaml.full_load(questions)
            question = ""
            for i, j in q.items():
                question += f"{j[r_r]} \n"
        await message.answer(question)
        await NewQuestion.Answer.set()
        await state.update_data(item=item)

@dp.message_handler(state=NewQuestion.Answer)
async def answer(message: types.Message, state: FSMContext):
    user_answer = message.text
    data = await state.get_data()
    item: Test = data.get("item")
    item.user_answer = user_answer
    t1 = f" {item.user_answer} "
    t2 = f" {item.right_answer} "

    if t1 not in t2:
        item.result = False
    else:
        item.result = True
        answers = await db.check_answers()
        await db.update_answers(answers+1)
    await state.update_data(item=item)
    await item.create()
    questions = await db.check_questions()
    await db.update_questions(questions+1)
    group_id = await db.check_group_id()
    if group_id:
        labs = await db.check_group_labs()
        await db.update_labs(labs)
    await message.answer("Для прожолжения нажмите /enter или введите любой текст. Для отмены нажмите /cancel.")
    await NewQuestion.Question.set()

@dp.message_handler(commands=["members"])
async def check_members(message: types.Message):
    members = await db.check_members()
    text = ("Участники Вашей группы:\n{members}").format(members=members)
    await message.answer(text)

@dp.message_handler(commands=["labs"])
async def lab(message: types.Message):
    await message.answer("Загрузите .pickle-файл для проверки лабораторной работы. Для отмены нажмите /cancel.")
    await NewLab.Lab.set()

@dp.message_handler(content_types=['document'], state=NewLab.Lab)
async def result(message: types.Message, state: FSMContext):
    import urllib
    from config import TOKEN
    user = types.User.get_current()
    user_id = user.id
    document_id = message.document.file_id
    file_info = await bot.get_file(document_id)
    fi = file_info.file_path
    name = message.document.file_name

    ext = f" {name} "
    check = f".pickle"
    if check not in ext:
        await message.answer("Файл должен быть разрешения .pickle!\n")
        await message.answer("Проверка отменена. Для повторной попытки нажмите /labs.")
        await state.reset_state()
    else:
        urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{TOKEN}/{user_id}/{fi}', f'./{name}')
        await bot.send_message(message.from_user.id, 'Файл успешно сохранён!')

    lab_number = 1  # Номер лабораторной работы
    lab_result = 1  # Результат лабораторной работы

    row1 = await db.raw1()
    row2 = await db.raw2()
    if row1:
        for i in row1:
            for j in row2:
                if int(i[0]) == lab_number and j[0] == True:
                    await message.answer(f"Вы уже выполнили эту лабораторную работу!")
                    await state.reset_state()
                else:
                    item = Lab()
                    dt = datetime.now()
                    group_id = await db.check_id()
                    item.user_id = user.id
                    item.group_id = group_id
                    item.lab_number = lab_number
                    if lab_result == 1:
                        item.result = True
                        labs_done = await db.check_labs()
                        await db.update_labs(labs_done + 1)
                        labs = await db.check_labs()
                        await db.update_group_labs(labs)
                        await message.answer(f"Лабораторная работа №{lab_number} выполнена верно!")
                    else:
                        await message.answer(f"Лабораторная работа №{lab_number} выполнена неверно!")
                    item.date_time = dt
                    await state.update_data(item=item)
                    await item.create()
                    await state.reset_state()
    else:
        item = Lab()
        dt = datetime.now()
        group_id = await db.check_id()
        item.user_id = user.id
        item.group_id = group_id
        item.lab_number = lab_number
        if lab_result == 1:
            item.result = True
            labs_done = await db.check_labs()
            await db.update_labs(labs_done + 1)
            labs = await db.check_labs()
            await db.update_group_labs(labs)
            await message.answer(f"Лабораторная работа №{lab_number} выполнена верно!")
        else:
            await message.answer(f"Лабораторная работа №{lab_number} выполнена неверно!")
        item.date_time = dt
        await state.update_data(item=item)
        await item.create()
        await state.reset_state()

@dp.message_handler()
async def other_echo(message: Message):
    await message.answer(message.text)

