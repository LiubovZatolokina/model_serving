FROM pytorch/torchserve:latest
USER root
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y python3-distutils
RUN apt-get install -y openjdk-11-jdk
COPY ./ /home/model-server/
RUN chmod -R a+rw /home/model-server/
USER model-server
RUN pip3 install --upgrade pip
RUN pip install torch-model-archiver
RUN pip install -r /home/model-server/requirements.txt

RUN torch-model-archiver --model-name lstm_attention \
--model-file /home/model-server/model.py \
--version 1.0 --serialized-file /home/model-server/lstm_attention.pt \
--handler /home/model-server/handler.py \
--extra-files /home/model-server/label_obj.pt,/home/model-server/source_vocab.pt,/home/model-server/lstm_attention.pt
RUN mv lstm_attention.mar deployment/lstm_attention.mar
CMD [ "torchserve", "--start", "--model-store", "deployment/", "--models", "lstm_attention.mar" ]