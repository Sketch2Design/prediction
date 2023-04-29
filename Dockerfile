FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app