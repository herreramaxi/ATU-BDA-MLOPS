#mh v2.2
FROM python:3.8
EXPOSE 5001
WORKDIR /opt/app
COPY . /opt/app
RUN pip3.8 install -r requirements.txt
RUN python3.8 model.py
CMD ["python3.8", "/opt/app/flaskApp.py"]
