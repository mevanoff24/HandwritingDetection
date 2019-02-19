FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev build-essential python-opencv git zip unzip \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install awscli aws tensorflow==1.12.0

RUN pip install --upgrade awscli==1.14.5 s3cmd==2.0.1 python-magic

RUN cd /tmp \
  && git clone https://github.com/mevanoff24/HandwritingDetection.git \
  && cd HandwritingDetection \
  && cd build \
  && chmod +x ./environment.sh \
  && chmod +x ./beam_search.sh \
  && cd app/models/context2vec \
  && mkdir models_103 \
  && cd models_103 \
  && aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-2/embedding.vec . \
  && aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-2/model.param . \
  && aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-2/model.param.config.json . \
  && aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-2/model.param.optim .

RUN cd /tmp/HandwritingDetection/build/app/models/ocr \
  && mkdir models \
  && cd models \
  && aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/ocr/weights-improvement2-10-01-3.00.hdf5 .

 
RUN cd /tmp/HandwritingDetection/build/app/models/OCRBeamSearch/src \
  && git clone https://github.com/githubharald/CTCWordBeamSearch.git \
  && cd CTCWordBeamSearch/cpp/proj \
  && cp /tmp/HandwritingDetection/build/beam_search.sh . \
  && ./beam_search.sh
  
  
RUN cd /tmp/HandwritingDetection/build/app/models/OCRBeamSearch/src \
  && cp CTCWordBeamSearch/cpp/proj/TFWordBeamSearch.so . \
  && rm -rf CTCWordBeamSearch
    
RUN cd /tmp/HandwritingDetection/build/app/models/OCRBeamSearch/model \
  && rm checkpoint \
  && unzip model.zip  
  

COPY . .
WORKDIR /tmp/HandwritingDetection/build/app
RUN pip3 install -r ../requirements.txt
RUN python -m nltk.downloader wordnet
ENTRYPOINT ["python3"]
CMD ["run.py"]

