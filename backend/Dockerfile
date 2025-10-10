FROM python:3.13-slim
WORKDIR /backend
COPY requirements.txt /backend
RUN pip3 install --upgrade pip -r requirements.txt
COPY . /backend
