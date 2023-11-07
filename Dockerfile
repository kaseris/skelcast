FROM pytorch/pytorch:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install --editable .

ENTRYPOINT [ "main.py" ]
