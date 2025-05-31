FROM python:3.8
EXPOSE 5001
WORKDIR /opt/app

ARG BUILD
ARG ENVIRONMENT

ENV BUILD=$BUILD
ENV APP_ENV=$ENVIRONMENT

# Copy app files
COPY flaskApp.py /opt/app/
COPY SpamModel.pkl /opt/app/
COPY requirements.txt /opt/app/
COPY templates/ /opt/app/templates/

RUN pip3.8 install -r requirements.txt
CMD ["python3.8", "flaskApp.py"]