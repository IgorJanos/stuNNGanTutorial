# syntax=docker/dockerfile:1
FROM vggfiit/vgg-torch:latest

# Update the APT-GET
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install ca-certificates
RUN update-ca-certificates


USER root

# Install required packages
WORKDIR /setup
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/stuNNGanTutorial