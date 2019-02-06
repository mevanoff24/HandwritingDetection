# download Language Model
cd ~/HandwritingDetection/build/app/models/context2vec/
mkdir models
cd models/
aws s3 cp s3://handwrittingdetection/models/language_model/embedding.vec .
aws s3 cp s3://handwrittingdetection/models/language_model/model.param .
aws s3 cp s3://handwrittingdetection/models/language_model/model.param.config.json .
aws s3 cp s3://handwrittingdetection/models/language_model/model.param.optim .

# download OCR Model
cd ~/HandwritingDetection/build/app/models/ocr/
mkdir models
cd models/
aws s3 cp s3://handwrittingdetection/models/ocr/weights-improvement2-10-01-3.00.hdf5 .

# download sample image
cd ~/HandwritingDetection/data
mkdir samples
cd samples/
aws s3 cp s3://handwrittingdetection/data/word_level/c03/c03-096f/c03-096f-03-05.png .