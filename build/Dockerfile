FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev build-essential python-opencv git \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN cd /tmp \
  && git clone https://github.com/mevanoff24/HandwritingDetection.git \
  && cd HandwritingDetection \
  && cd build \
  && chmod +x ./environment.sh \
  && chmod +x ./beam_search.sh

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader wordnet
ENTRYPOINT ["python3"]
CMD ["app/run.py"]

