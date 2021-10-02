FROM ubuntu:18.04 AS release

#RUN yes | apt install software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN yes | apt install python3.9
#RUN yes | apt-get -y install python3-pip
#RUN pip3 install python-telegram-bot pythonping pyyaml aiogram python-dotenv gino sqlalchemy asyncpg sklearn pandas

ENV PGVER 10
RUN apt -y update && apt install -y postgresql-$PGVER
USER postgres
# COPY . .
# RUN cat ./postgresql.conf >> /etc/postgresql/$PGVER/main/postgresql.conf
RUN /etc/init.d/postgresql start &&\
    psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" &&\
    createdb -O docker myService &&\
    /etc/init.d/postgresql stop

RUN echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/$PGVER/main/pg_hba.conf

# PostgreSQL port
EXPOSE 5432

# config, logs and databases
#VOLUME  ["/etc/postgresql", "/var/log/postgresql", "/var/lib/postgresql"]

RUN service postgresql start && service postgresql status

# подключение образа python
# FROM python

# установка пакетов
USER root

RUN yes | apt install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN yes | apt-get install build-essential libssl-dev libffi-dev python-dev python3-dev apt-utils
RUN yes | apt install python3.7
# RUN python3 -V
RUN yes | apt-get -y install python3-pip
RUN pip3 install python-telegram-bot pythonping pyyaml python-dotenv gino sqlalchemy asyncpg sklearn pandas
RUN python3.7 -m pip install aiogram
#RUN pip3 install -U aiogram

# назначение текущей папки
WORKDIR /code

# копирование кода
COPY . /code

RUN python3.7 -V

# утилита, вызывамая при запуске
ENTRYPOINT ["python3.7"]

# файла для запуска
CMD [ "code/telegram/app.py" ]