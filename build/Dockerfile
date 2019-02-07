FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev build-essential python-opencv \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

CMD ["sh", "environment.sh"]
CMD ["sh", "beam_search.sh"]

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader wordnet
ENTRYPOINT ["python3"]
CMD ["app/run.py"]

