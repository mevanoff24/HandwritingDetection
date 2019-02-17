# Decipher


## Overview

Dockerized and AWS hosted Flask app to decipher messy handwriting to predict most likely word choice. You can navigate to the url [here](bit.ly/decipherAI) to use the deployed application. 


## Motivation for this project
Have you ever read handwritten text when you came across an indecipherable word? This is a big issue in pharmacies mis-prescribing medicine, maintenance workers mis-communicating results, or even reading lecture notes. The use cases for predicting messy handwriting is far and wide. 


## Solution
I have utilized an [Optical Character Recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) and [context2vec](https://u.cs.biu.ac.il/~melamuo/publications/context2vec_conll16.pdf) models with a custom weighing algorithm results from each model to decipher messy handwriting to predict most likely text. 

## Build Environment

## Docker Setup
If you have [docker](https://www.docker.com/) set up on your system follow these simple steps to deploy the app
**Steps**
1. clone this repo
```
https://github.com/mevanoff24/HandwritingDetection.git
```
2. In the root directory of `HandwritingDetection`, build the docker image using 
```
docker-compose build
```
3. Start the application by running
```
docker-compose up
```
4. Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the application. 


## Non-Docker Setup
1. clone this repo
```
https://github.com/mevanoff24/HandwritingDetection.git
```
2. Navigate to the `HandwritingDetection/build` with `cd cd HandwritingDetection/build/` and install all requirement packages 
```
pip install -r requiremnts
```
3. Optionally, download the data from [S3](https://aws.amazon.com/s3/) by running 
```
sh environment.sh
```
4. To compile beam search from tensorflow and unzip OCR models run the command
```
bash ./beam_search_local.sh
```




#### Dependencies
```
Flask
torch
tensorflow
numpy
pandas
nltk
boto3
opencv-python
toml
editdistance
python-Levenshtein
```



## Build Models Locally

### Context2vec
1. Navigate to `HandwritingDetection/build/app/models/context2vec`
2. The most basic way to start training is by running
```
python main.py -t TRAINING FILE
```
Where `-t` expects the training file. The `main` module expects your input to be a list of lists where each list is one example sentence, phrase or short paragraph. You may also pass in an optional validation set with the `-v`. Or you may pull data from a S3 bucket by using the `-s True` flag. 

More optional flags available. See `--help`. 



## Example

Input:
```
In: X = 'We' + ' [] ' + 'in the house'
In: Image.open(file_url)
```
OCR Model Prediction
```
In: ocr_pred, ocr_prob = inference_model.run_beam_ocr_inference_by_user_image(file_url)
In: print('OCR prediction is "{}" with probability of {}%'.format(ocr_pred[0], round(ocr_prob[0]*100)))
```
```
Out: OCR prediction is "like" with probability of 83.0%
```
Language Model Prediction
```
In: lm_preds = inference_model.run_lm_inference_by_user_input(X, topK=10)
In: print('Top 10 LM predictions: {}'.format([w for _, w in lm_preds]))
```
```
Out: Top 10 LM predictions: ['slept', 'dabble', "'re", 'stayed', 'sat', 'lived', 'hid', 'got', 'live']
```
Weighed Model
```
In: features = inference_model.create_features_improved(lm_preds, ocr_pred, ocr_prob)
In: inference_model.final_scores(features, ocr_pred, ocr_prob_threshold=0.85, return_topK=10)
```
```
Out: 
[('live', 4.8623097696683555),
 ('lived', 3.448472232239753),
 ('dabble', 3.00382016921238),
 ("'re", 2.888073804708552),
 ('slept', 2.5013190095196265),
 ('hid', 2.161875374647212),
 ('stayed', 1.9861207593784505),
 ('sat', 1.7082426527844938),
 ('got', 1.6237610856401)]
```



## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
