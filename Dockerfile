FROM ubuntu:latest

RUN apt-get update -y 
RUN apt-get install -y python3 python3-dev python3-pip
ADD requirements.txt /home/requirements.txt
WORKDIR /home
RUN pip3 install -r /home/requirements.txt

CMD ["/bin/bash"]

