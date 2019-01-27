
from flask import Flask, render_template, request, jsonify, session
from wtforms import Form, TextAreaField, validators

import sys
import os
import pandas as pd 
import numpy as np 

# from .config import *


import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


from models.all_models import main


# TODO config file
TMPDIR = 'tmp/'
PORT = 5000
TMP_IMG_NAME = 'uploaded.jpg'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = TMPDIR
app.config['ALLOWEDEXTENSIONS'] = ALLOWED_EXTENSIONS



photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
	photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
	submit = SubmitField(u'Upload')



class SubmissionFormLeft(Form):
	left_text_form = TextAreaField('', 
					[validators.DataRequired(),
					validators.length(min = 1)])


class SubmissionFormRight(Form):
	right_text_form = TextAreaField('', 
					[validators.DataRequired(),
					validators.length(min = 1)])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	form = UploadForm()
	left_word_form = SubmissionFormLeft(request.form)
	right_word_form = SubmissionFormRight(request.form)
	if form.validate_on_submit():
		if not os.path.exists(TMPDIR):
			os.makedirs(TMPDIR)
		# save_img_path = os.path.join(TMPDIR, 'uploaded.jpg')
		filename = photos.save(form.photo.data, name=TMP_IMG_NAME)
		file_url = photos.url(filename)
	else:
		file_url = None
		filename = None
	session['file_url'] = file_url
	session['filename'] = filename
	return render_template('index.html', form=form, left_word_form=left_word_form, 
							right_word_form=right_word_form, file_url=file_url)



# @app.route('/')
# def index():
# 	form = SubmissionForm(request.form)
# 	return render_template('index.html', form=form)

@app.route('/main', methods=['GET', 'POST'])
def main():
	return render_template('main.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
	file_url = session.get('file_url', None)
	filename = session.get('filename', None)
	file_url = 'tmp/' + str(file_url)
	filename = 'tmp/' + str(filename)
	# print(os.getcwd())
	form = UploadForm()
	left_word_form = SubmissionFormLeft(request.form)
	right_word_form = SubmissionFormRight(request.form)
	if request.method == 'POST' and left_word_form.validate() and right_word_form.validate():
		left_text = request.form['left_text_form']
		right_text = request.form['right_text_form']
		# file_url = photos.url(filename)
		# print(left_text, right_text, file_url, filename)
	# print(file_url)
		ocr_pred, len_pred, lm_pred = main(left_text, right_text, filename)
	# print(filename)
	# print(ocr_pred, len_pred, lm_pred)

	return render_template('predict.html', left_text=left_text, right_text=right_text, file_url=filename, 
							ocr_pred=ocr_pred, len_pred=len_pred, lm_pred=lm_pred)




if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)


