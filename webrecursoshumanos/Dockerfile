FROM python:3-slim
WORKDIR /webrecursoshumandos
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y python3-dev && \
    apt-get install -y libatlas-base-dev && \
    apt-get install -y gfortran
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python3", "./app.py" ]

