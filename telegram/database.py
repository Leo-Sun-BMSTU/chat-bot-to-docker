from aiogram import types, Bot
from gino import Gino
from gino.schema import GinoSchemaVisitor
from sqlalchemy import (Column, Integer, BigInteger, String,
                        Sequence, TIMESTAMP, Boolean)
from sqlalchemy import sql
from datetime import datetime

from config import db_pass, db_user, host

db = Gino()


class User(db.Model):
    __tablename__ = 'users'

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    user_id = Column(BigInteger)
    full_name = Column(String(100))
    username = Column(String(50))
    group_id = Column(Integer)
    labs_done = Column(Integer, default=0)
    questions_given = Column(Integer, default=0)
    right_answers = Column(Integer, default=0)
    date_time = Column(TIMESTAMP)
    query: sql.Select

    def __repr__(self):
        return "<User(id='{}', fullname='{}', username='{}')>".format(
            self.id, self.full_name, self.username)


class Lab(db.Model):
    __tablename__ = 'labs'
    query: sql.Select

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    user_id = Column(BigInteger)
    group_id = Column(Integer)
    lab_number = Column(Integer)
    result = Column(Boolean, default=False)
    date_time = Column(TIMESTAMP)

    def __repr__(self):
        return "<Item(id='{}', name='{}', price='{}')>".format(
            self.id, self.name, self.price)


class Test(db.Model):
    __tablename__ = 'tests'
    query: sql.Select

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    user_id = Column(BigInteger)
    question = Column(Integer)
    right_answer = Column(String(250))
    user_answer = Column(String(250))
    result = Column(Boolean, default=False)
    date_time = Column(TIMESTAMP)


class DBCommands:

    async def get_user(self, user_id) -> User:
        user = await User.query.where(User.user_id == user_id).gino.first()
        return user

    async def add_new_user(self, group_id=None) -> User:
        user = types.User.get_current()
        old_user = await self.get_user(user.id)
        if old_user:
            return old_user
        dt = datetime.now()
        new_user = User()
        new_user.user_id = user.id
        new_user.username = user.username
        new_user.full_name = user.full_name
        new_user.date_time = dt

        if group_id:
            new_user.group_id = int(group_id)
        await new_user.create()
        return new_user

    async def count_users(self) -> int:
        total = await db.func.count(User.id).gino.scalar()
        return total

    async def check_members(self):
        bot = Bot.get_current()
        user_id = types.User.get_current().id

        user = await User.query.where(User.user_id == user_id).gino.first()
        members = await User.query.where(User.group_id == user.id).gino.all()

        return ", ".join([
            f"{num + 1}. " + (await bot.get_chat(group_id.user_id)).get_mention(as_html=True)
            for num, group_id in enumerate(members)
        ])

    async def check_questions(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        questions_given = await User.select('questions_given').where(User.id == user.id).gino.first()
        for questions, empty in enumerate(questions_given):
            return empty

    async def check_answers(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        right_answers = await User.select('right_answers').where(User.id == user.id).gino.first()
        for answers, empty in enumerate(right_answers):
            return empty

    async def check_labs(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        labs_done = await User.select('labs_done').where(User.id == user.id).gino.first()
        for labs, empty in enumerate(labs_done):
            return empty

    async def check_id(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        group_id = await User.select('id').where(User.id == user.id).gino.first()
        for some, empty in enumerate(group_id):
            return empty

    async def check_group_id(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        group_id = await User.select('group_id').where(User.id == user.id).gino.first()
        for some, empty in enumerate(group_id):
            return empty

    async def check_group_labs(self):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        group_labs = await User.select('labs_done').where(User.id == user.group_id).gino.first()
        for labs, empty in enumerate(group_labs):
            return empty

    async def update_questions(self, questions):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        questions_given = await User.update.values(questions_given=questions).where(User.id == user.id).gino.status()
        return questions_given

    async def update_answers(self, answers):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        right_answers = await User.update.values(right_answers=answers).where(User.id == user.id).gino.status()
        return right_answers

    async def update_labs(self, labs_done):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        labs_done = await User.update.values(labs_done=labs_done).where(User.id == user.id).gino.status()
        return labs_done

    async def update_group_labs(self, labs):
        user_id = types.User.get_current().id
        user = await User.query.where(User.user_id == user_id).gino.first()
        labs = await User.update.values(labs_done=labs).where(User.group_id == user.id).gino.status()
        return labs

    async def row(self):
        user_id = types.User.get_current().id
        user = await Test.query.where(Test.user_id == user_id).gino.first()
        row = await Test.select('question').where(Test.user_id == user_id).gino.all()
        return row

    async def raw1(self):
        user_id = types.User.get_current().id
        user = await Lab.query.where(Lab.user_id == user_id).gino.first()
        raw1 = await Lab.select('lab_number').where(Lab.user_id == user_id).gino.all()
        return raw1

    async def raw2(self):
        user_id = types.User.get_current().id
        user = await Lab.query.where(Lab.user_id == user_id).gino.first()
        raw2 = await Lab.select('result').where(Lab.user_id == user_id).gino.all()
        return raw2


async def create_db():
    await db.set_bind(f'postgresql://{db_user}:{db_pass}@{host}/gino')

    # Create tables
    db.gino: GinoSchemaVisitor
    await db.gino.drop_all()
    await db.gino.create_all()
