# make an image of the project 

FROM python:3.8.5-slim-buster

# copy the dependencies file to the working directory
COPY  . /app

# set the working directory in the container
WORKDIR /app


# install dependencies
RUN pip install -r requirements.txt

COPY models/scratch_model.h5 models/
COPY models/pretrained_ResNet152V2.h5 models/

EXPOSE 5000

# command to run on container start
CMD python app.py

# docker build -t myimage .


