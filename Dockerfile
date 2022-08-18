FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip intall -r requirements.txt
COPY . . 
ENTRYPOINT "bash"
CMD ["ls"]