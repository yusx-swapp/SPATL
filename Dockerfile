FROM python:3.9-slim-buster

COPY /requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt


WORKDIR /app

COPY . .
