# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV SERVICE_ACCOUNT_JSON=application_default_credentials.json

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
