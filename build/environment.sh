# download Language Model
cd ~/HandwritingDetection/build/app/models/context2vec/
mkdir models_103
cd models_103/
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-103/embedding.vec .
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-103/model.param .
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-103/model.param.config.json .
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/language_model/wiki-103/model.param.optim .

# download OCR Model
cd ~/HandwritingDetection/build/app/models/ocr/
mkdir models
cd models/
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/models/ocr/weights-improvement2-10-01-3.00.hdf5 .

# download sample image
cd ~/HandwritingDetection/data
mkdir samples
cd samples/
aws --no-sign-request s3 --region=us-west-2 cp s3://handwrittingdetection/data/word_level/c03/c03-096f/c03-096f-03-05.png .
