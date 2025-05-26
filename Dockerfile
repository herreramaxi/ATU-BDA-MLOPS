FROM python:3.8
EXPOSE 5001
WORKDIR /opt/app
COPY . /opt/app
RUN pip3.8 install -r requirements.txt
CMD ["python3.8", "flaskApp.py"]