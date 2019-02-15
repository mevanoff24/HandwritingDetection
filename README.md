# Decipher


## Overview

Dockerized and AWS hosted Flask app to decipher messy handwriting to predict most likely word choice. You can navigate to the url [here](bit.ly/decipherAI) to use the deployed application. 


## Motivation for this project:
Have you ever read handwritten text when you came across an indecipherable word? This is a big issue in pharmacies mis-prescribing medicine, maintenance workers mis-communicating results, or even reading lecture notes. The use cases are far and wide. 


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
2. In the root directory of `HandwritingDetection` first install all requirement packages 
```
pip install -r requiremnts
```
3. After `cd`ing into the `build` directory, optionally, download the data from [S3](https://aws.amazon.com/s3/) by running 
```
sh environment.sh
```
4. To compile beam search from tensorflow run the command
```
sh beam_search.sh
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
#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```




## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
