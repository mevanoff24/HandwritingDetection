# Decipher

<img src="build/app/static/images/detective.jpeg" alt="drawing" width="100"/>



## Overview

Dockerized and AWS hosted Flask app to decipher messy handwriting to predict most likely word choice. Please [contact me](mailto:mevanoff24@gmail.com) if you need to use the deployed application, since the EC2 instance is currently down.  

You can see my presentation [here](https://docs.google.com/presentation/d/1zYfLiZooCKe1LT3e8FkTNU39ZUZoNQfaUbTzMXQ1yn4/edit#slide=id.p)


## Motivation for this project
Have you ever read handwritten text when you came across an indecipherable word? This is a big issue in pharmacies mis-prescribing medicine, maintenance workers mis-communicating results, or even reading lecture notes. The use cases for predicting messy handwriting is far and wide. 


## Solution
I have utilized an [Optical Character Recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) and [context2vec](https://u.cs.biu.ac.il/~melamuo/publications/context2vec_conll16.pdf) models with a custom weighing algorithm results from each model to decipher messy handwriting to predict most likely text. 


## Pipeline 

![gif](static/pipeGIF.gif)


## Example


Input:
```
In: file_url = 'data/samples/c03-096f-07-05'
In: X = 'We' + ' [] ' + 'in the house'
In: print(X)
In: Image.open(file_url)
```
```
Out: We [] in the house
```
![c03-096f-07-05](https://user-images.githubusercontent.com/8717434/52908809-ec414e80-3231-11e9-8dc8-af13f6451960.png)


**OCR Model Prediction**
```
In: ocr_pred, ocr_prob = inference_model.run_beam_ocr_inference_by_user_image(file_url)
In: print('OCR prediction is "{}" with probability of {}%'.format(ocr_pred[0], round(ocr_prob[0]*100)))
```
```
Out: OCR prediction is "like" with probability of 83.0%
```
**Language Model Prediction**
```
In: lm_preds = inference_model.run_lm_inference_by_user_input(X, topK=10)
In: print('Top 10 LM predictions: {}'.format([w for _, w in lm_preds]))
```
```
Out: Top 10 LM predictions: ['slept', 'dabble', "'re", 'stayed', 'sat', 'lived', 'hid', 'got', 'live']
```
**Weighed Algorithm**
```
In: features = inference_model.create_features_improved(lm_preds, ocr_pred, ocr_prob)
In: inference_model.final_scores(features, ocr_pred, ocr_prob_threshold=0.85, return_topK=10)
```
```
Out: 
[('live', 4.8623097696683555), <---- Final prediction
 ('lived', 3.448472232239753),
 ('dabble', 3.00382016921238),
 ("'re", 2.888073804708552),
 ('slept', 2.5013190095196265),
 ('hid', 2.161875374647212),
 ('stayed', 1.9861207593784505),
 ('sat', 1.7082426527844938),
 ('got', 1.6237610856401)]
```

As you can see above, the initial OCR model predicted this image incorrectly. Predicted "like" instead of "live". While the LM model had the 'correct' answer in the topK list. We then can create 'features' and create a new Weighed Algorithm to be able to correctly classify this image as "live". 

-----

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
2. Navigate to the `HandwritingDetection/build` with `cd HandwritingDetection/build/` and install all requirement packages 
```
pip install -r requirements.txt
```
3. Optionally, download the data from [S3](https://aws.amazon.com/s3/) by running 
```
sh environment.sh
```
4. To compile beam search from tensorflow and unzip OCR models run the command
```
bash ./beam_search_local.sh
```
5. You then can go into the `app` directory (`cd app/`) and run
```
python run.py
```
To start the Flask server at [http://0.0.0.0:5000/](http://0.0.0.0:5000/). Or just play around with the repo. 

6. This repo also contains a couple of sample images under the `data/samples` directory to upload to the Flask app. 



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
You can install all requirement packages from this root directory with the command
```
pip install -r build/requirements.txt
```

## Data

[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- The IAM Handwriting database is the biggest database of English handwriting images. It has 1539 pages of scanned text written by 600+ writers.

[WikiText2 and WikiText103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
- The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.


-----

## Build Models Locally

### Context2vec
1. Navigate to `HandwritingDetection/build/app/models/context2vec`
2. The most basic way to start training is by running
```
python main.py -t TRAINING FILE
```
Where `-t` expects the training file. The `main` module expects your input to be a list of lists where each list is one example sentence, phrase or short paragraph. You may also pass in an optional validation set with the `-v`. Or you may pull data from a S3 bucket by using the `-s true` flag. The easiest way to run the model is to pull data from the S3 bucket with the command
```
python main.py -s true
```
More optional flags available. See `--help`. 


### Optical Character Recognition Model 

I use the IAM dataset. To get the dataset:

1. Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
2. Download `words/words.tgz`.
3. Create the directory `data/raw/word_level/`.
4. Put the content (directories a01, a02, ...) of `words.tgz` into `data/raw/word_level/`.

To train the model, navigate to the directory `HandwritingDetection/build/app/models/OCRBeamSearch/src` and run 
```
python main.py --train --uses3
```

-----

## Results

| Model  | DataSet  |  Accuracy |  Stem Accuracy | Word Vector Similarity  |
|---|---|---|---|---|
| Individual Language Model  | Wiki-103  | 0.260  | 0.263  |  4.915 |
|  Individual OCR Beam Search | Wiki-103  | 0.908  | 0.912  | 0.677  |
| Weighted LM + OCR Beam Search  | Wiki-103  | 0.911  | 0.916  | 0.616  |


-----


## Content 
This section overviews how the repo is built (i.e. the folder structure)

The most important directory is the`models` directory (`build/app/models`) where each individual model lives
- Language Model -- `context2vec`
- Optical Character Recognition Beam Search -- `OCRBeamSearch`
This is where all the training takes place

Inference takes place in the `inference.py` file (`build/app/inference.py`)

```bash
├── build
│   ├── app
│   │   ├── config.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── models
│   │   │   ├── all_models.py
│   │   │   ├── context2vec
│   │   │   │   ├── config.toml
│   │   │   │   ├── __init__.py
│   │   │   │   ├── logs
│   │   │   │   │   └── logs.txt
│   │   │   │   ├── main.py
│   │   │   │   └── src
│   │   │   │       ├── args.py
│   │   │   │       ├── config.py
│   │   │   │       ├── dataset.py
│   │   │   │       ├── __init__.py
│   │   │   │       ├── model.py
│   │   │   │       ├── mscc_eval.py
│   │   │   │       ├── negative_sampling.py
│   │   │   │       ├── utils.py
│   │   │   │       └── walker_alias.py
│   │   │   ├── __init__.py
│   │   │   ├── ocr
│   │   │   │   └── src
│   │   │   │       ├── args.py
│   │   │   │       ├── config.py
│   │   │   │       ├── generator.py
│   │   │   │       ├── ocr_model.py
│   │   │   │       └── spellchecker.py
│   │   │   └── OCRBeamSearch
│   │   │       ├── data
│   │   │       │   ├── analyze.png
│   │   │       │   ├── checkDirs.py
│   │   │       │   ├── corpus.txt
│   │   │       │   ├── Get IAM training data.txt
│   │   │       │   ├── pixelRelevance.npy
│   │   │       │   ├── test.png
│   │   │       │   ├── translationInvariance.npy
│   │   │       │   ├── translationInvarianceTexts.pickle
│   │   │       │   ├── wiki2.txt
│   │   │       │   └── words.txt
│   │   │       ├── LICENSE.md
│   │   │       ├── model
│   │   │       │   ├── accuracy.txt
│   │   │       │   ├── charList.txt
│   │   │       │   ├── checkpoint
│   │   │       │   ├── model.zip
│   │   │       │   └── wordCharList.txt
│   │   │       ├── model_new
│   │   │       │   ├── accuracy.txt
│   │   │       │   ├── charList.txt
│   │   │       │   └── wordCharList.txt
│   │   │       └── src
│   │   │           ├── main.py
│   │   │           ├── Model.py
│   │   │           ├── NewDataLoader.py
│   │   │           ├── SamplePreprocessor.py
│   │   │           └── TFWordBeamSearch.so
│   │   ├── run.py
│   │   ├── static
│   │   │   ├── css
│   │   │   │   ├── bootstrap.css
│   │   │   │   └── my_css.css
│   │   │   ├── images
│   │   │   │   └── detective.jpeg
│   │   │   └── js
│   │   │       ├── bootstrap.bundle.js
│   │   │       └── bootstrap.js
│   │   ├── templates
│   │   │   ├── add_image.html
│   │   │   ├── _form_helpers.html
│   │   │   └── predict.html
│   │   └── utils.py
│   ├── beam_search_install.ipynb
│   ├── beam_search_local.sh
│   ├── beam_search.sh
│   ├── data_processing
│   │   ├── image_meta.py
│   │   ├── wiki_data.py
│   │   └── word_level.py
│   ├── Dockerfile
│   ├── environment.sh
│   ├── notebooks
│   │   ├── DatasetCreation.ipynb
│   │   ├── Evaluation.ipynb
│   │   ├── FullMeta.ipynb
│   │   ├── keras.ipynb
│   │   ├── LM_model.ipynb
│   │   ├── Meta.ipynb
│   │   ├── NewOCR.ipynb
│   │   ├── OCR_model.ipynb
│   │   ├── Pipeline.ipynb
│   │   ├── __pycache__
│   │   ├── s3_OCR.ipynb
│   │   ├── tensract.ipynb
│   │   ├── visuals.py
│   │   └── wiki_dataset.ipynb
│   └── requirements.txt
├── configs
│   └── example_config.yml
├── data
│   ├── preprocessed
│   │   ├── example.txt
│   │   ├── meta.csv
│   │   ├── meta.json
│   │   ├── meta_json.csv
│   │   ├── meta_json.json
│   │   ├── word_level_meta.csv
│   │   ├── word_level_test.csv
│   │   └── word_level_train.csv
│   ├── processed
│   │   └── example_output.txt
│   └── samples
│       ├── c03-096f-03-05.png
│       └── c03-096f-07-05.png
├── docker-compose.yml
├── LICENSE
├── README.md
├── static
│   └── pipeGIF.gif
└── tests
    └── README.md
```



-----


### Acknowledgements

Big thank you to Harald Scheidl (githubharald) and his [SimpleHTR](https://github.com/githubharald/SimpleHTR) implementation of his Handwritten Text Recognition (HTR) system and [CTC Word Beam Search Decoding Algorithm](https://github.com/githubharald/CTCWordBeamSearch). His Beam Search implementation saved me a lot of time in this short 3-4 week project. 
