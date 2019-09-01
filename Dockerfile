FROM ubuntu:latest

# RUN apk add --no-cache python3-dev \
#     && pip3 install --upgrade pip

COPY . /app

WORKDIR /app

# RUN pip install -r requirements.txt

EXPOSE 5000

# CMD python ./Server.py