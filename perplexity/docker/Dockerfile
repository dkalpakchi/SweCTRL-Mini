FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN apt-get update && apt-get install -y python3-pip locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN useradd -d /home/user -m user
USER user
WORKDIR /home/user

RUN pip install transformers --user
RUN pip install scipy numpy tqdm sentencepiece --user
RUN pip install jsonlines udon2 --user
