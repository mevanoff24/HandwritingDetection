from flask import Flask, render_template, request, jsonify, session
from wtforms import Form, TextAreaField, validators
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import sys
import os
import json
import uuid

from config import *
from inference import Inference


application = app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = TMPDIR
app.config['ALLOWEDEXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = 'static/Uploads'


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# load Inference class
inference_model = Inference(device='cpu', decoding='wordbeamsearch')

class SubmissionFormLeft(Form):
	left_text_form = TextAreaField('')
class SubmissionFormRight(Form):
	right_text_form = TextAreaField('')
      


@app.route('/', methods=['GET', 'POST'])
def upload_file():
	# remove old images and session 
	session.clear()
	for the_file in os.listdir(app.config['UPLOAD_FOLDER']):
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)

	left_word_form = SubmissionFormLeft(request.form)
	right_word_form = SubmissionFormRight(request.form)
	return render_template('add_image.html', left_word_form=left_word_form, 
							right_word_form=right_word_form)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		file = request.files['file']
		extension = os.path.splitext(file.filename)[1]
		f_name = str(uuid.uuid4()) + extension
		save_path = os.path.join(app.config['UPLOAD_FOLDER'], f_name)
		file.save(save_path)
		session['file_url'] = save_path
	return json.dumps({'filename':f_name})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file_url = session.get('file_url', None)
    left_word_form = SubmissionFormLeft(request.form)
    right_word_form = SubmissionFormRight(request.form)
    if request.method == 'POST' and left_word_form.validate() and right_word_form.validate():
        left_text = request.form['left_text_form']
        right_text = request.form['right_text_form']       

        X = left_text + ' [] ' + right_text
        
        try:
            prediction = inference_model.predict(X, file_url, ocr_prob_threshold=0.10)
        except:
            prediction = 'NO INPUT. TRY AGAIN'
            
    return render_template('predict.html', left_text=left_text, right_text=right_text, file_url=file_url, 
                            prediction=prediction)





if __name__ == '__main__':
	app.run(host='0.0.0.0', port=PORT, debug=DEBUG)