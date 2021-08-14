import datetime
import random
import yaml
from datetime import datetime
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import CommandStart
from aiogram.types import Message
import database
from telegram.config import admin_id
from load_all import dp, bot
from states import NewName, NewQuestion, NewLab
from database import User, Lab, Test
from checking import check_the_lab

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
                "Ввести фамилию и имя: /name (обязательно)\n"
                "\n"
                "Пройти тестирование: /test").format(
            count_users=count_users,
        )
    else:
        text = ("Приветствую Вас!\n"
             "Количество пользователей в базе данных: {count_users}\n"
             "\n"
             "Ввести фамилию и имя: /name (обязательно)\n"
             "\n"
             "Реферальная ссылка Вашей группы: {bot_link}\n"
             "Предоставьте данную ссылку всем участникам Вашей группы для входа в чат, чтобы бот занёс вас в одну группу.\n"
             "Посмотреть участников Вашей группы: /members\n"
             "\n"
             "Проверить лабораторную работу: /lab\n"
             "Пройти тестирование: /test").format(
        count_users=count_users,
        bot_link=bot_link
        )
    if message.from_user.id == admin_id:
        text += ("\n"
                  "\nОпция для администратора: /somecommand")
    await bot.send_message(chat_id, text)


@dp.message_handler(commands=["members"])
async def check_members(message: types.Message):
    members = await db.check_members()
    text = ("Участники Вашей группы:\n{members}").format(members=members)
    await message.answer(text)


@dp.message_handler(commands=["cancel"], state=NewName)
async def cancel(message: types.Message, state: FSMContext):
    await message.answer("Вы отменили ввод данных. Для повторной попытки нажмите /name.")
    await state.reset_state()

@dp.message_handler(commands=["name"])
async def name(message: types.Message):
    name_exists = await db.check_name()
    if name_exists:
        await message.answer(f"Вы уже вводили свои данные. Вас зовут {name_exists}.")
    else:
        await message.answer(f"Введите Ваши фамилию и имя. Для отмены нажмите /cancel.")
        await NewName.Name.set()

@dp.message_handler(state=NewName.Name)
async def set_name(message: types.Message, state: FSMContext):
    real_name = message.text
    await db.update_name(real_name)
    await message.answer(f"Вы записаны в базу как {real_name}.")
    await state.reset_state()


@dp.message_handler(commands=["cancel"], state=NewQuestion)
async def cancel(message: types.Message, state: FSMContext):
    await message.answer("Вы отменили тестирование. Для повторной попытки нажмите /test.")
    await state.reset_state()

@dp.message_handler(commands=["test"])
async def test(message: types.Message):
    await message.answer("Для продолжения нажмите /enter или введите любой текст. Для отмены нажмите /cancel.")
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
        var = await db.get_question()
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
    await message.answer("Для продолжения нажмите /enter или введите любой текст. Для отмены нажмите /cancel.")
    await NewQuestion.Question.set()


@dp.message_handler(commands=["cancel"], state=NewLab)
async def cancel(message: types.Message, state: FSMContext):
    await message.answer("Вы отменили проверку. Для повторной попытки нажмите /lab.")
    await state.reset_state()

@dp.message_handler(commands=["lab"])
async def lab(message: types.Message):
    await message.answer("Введите номер лабораторной работы, которую хотите проверить. Для отмены нажмите /cancel.")
    await NewLab.Lab.set()

@dp.message_handler(state=NewLab.Lab)
async def check_lab(message: types.Message, state: FSMContext):
    import os.path
    user_answer = message.text
    user = types.User.get_current()
    real_name = await db.check_name()

    dir1 = f'C:/ginodb/labs/{real_name}/data1.csv'
    if os.path.isfile(dir1):
        pass
    else:
        await message.answer(
                "Отсутствует файл data1.csv! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
        await state.reset_state()
    dir2 = f'C:/ginodb/labs/{real_name}/data2.csv'
    if os.path.isfile(dir2):
        pass
    else:
        await message.answer(
                "Отсутствует файл data2.csv! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
        await state.reset_state()

    if user_answer == "1":
        dir3 = f'C:/ginodb/labs/{real_name}/lab1.pickle'
        os.path.isfile(dir3)
        if os.path.isfile(dir3):
            res = check_the_lab(dir1, dir2, dir3)
            await message.answer(res)
        else:
            await message.answer(
                "Отсутствует файл lab1.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()

    if user_answer == "2":
        dir3 = f'C:/ginodb/labs/{real_name}/lab2.pickle'
        os.path.isfile(dir3)
        if os.path.isfile(dir3):
            res = check_the_lab(dir1, dir2, dir3)
            await message.answer(res)
        else:
            await message.answer(
                "Отсутствует файл lab2.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()

    if user_answer == "3":
        dir3 = f'C:/ginodb/labs/{real_name}/lab3_l2.pickle'
        os.path.isfile(dir3)
        if os.path.isfile(dir3):
            res1 = check_the_lab(dir1, dir2, dir3)
            await message.answer(res1)
        else:
            await message.answer(
                "Отсутствует файл lab3_l1.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()
        dir4 = f'C:/ginodb/labs/{real_name}/lab3_l2.pickle'
        os.path.isfile(dir4)
        if os.path.isfile(dir4):
            res2 = check_the_lab(dir1, dir2, dir4)
            await message.answer(res2)
        else:
            await message.answer(
                "Отсутствует файл lab3_l2.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()

    if user_answer == "4":
        dir3 = f'C:/ginodb/labs/{real_name}/lab4_RF.pickle'
        os.path.isfile(dir3)
        if os.path.isfile(dir3):
            res1 = check_the_lab(dir1, dir2, dir3)
            await message.answer(res1)
        else:
            await message.answer(
                "Отсутствует файл lab4_RF.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()
        dir4 = f'C:/ginodb/labs/{real_name}/lab4_CB.pickle'
        os.path.isfile(dir4)
        if os.path.isfile(dir4):
            res2 = check_the_lab(dir1, dir2, dir4)
            await message.answer(res2)
        else:
            await message.answer(
                "Отсутствует файл lab4_CB.pickle! Если Вы отправляли этот файл по почте, попробуйте произвести проверку позже. Файл ожидает загрузки.")
            await state.reset_state()


    lab_number = 1  # Номер лабораторной работы
    lab_result = 1  # Результат лабораторной работы


    number = await db.get_lab_number()
    result = await db.get_lab_result()
    if number:
        for i in number:
            for j in result:
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
                        await db.update_labs(labs_done+1)
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
            await db.update_labs(labs_done+1)
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

