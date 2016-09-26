#!/Users/yaoli/anaconda3/bin/python3

import os
import cv2
import numpy as np
from werkzeug import secure_filename
from flask import Flask, request, redirect, url_for, send_from_directory, render_template

from sklearn.externals import joblib

UPLOAD_FOLDER = '/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/website/upload'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

################################################################
# define a function to compute the input features for prediction
def compute_image_features(img_name, video_title):
    in_features = [] # [hue, sat, bri_global, dynamic_range, contrast_Mic, contrast_RMS, len_title]
    
    #####################################
    # use openCV compute
    # global hue, saturation, birightness (value)
    # dynamic range, Michelson contrast and RMS contrast
    try:
        img = cv2.imread(img_name)
    except Exception as e:
        print("Please upload an image!")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = img_hsv.mean(axis=(0,1))[0]/179
        in_features.append(hue)
        sat = img_hsv.mean(axis=(0,1))[1]/255
        in_features.append(sat)
        bri_global = img_hsv.mean(axis=(0,1))[2]/255 # value = brightness
        in_features.append(bri_global)
        #print("The global hue, saturation, brightness: ({0}, {1}, {2})".format(hue, sat, bri_global))
        
        bri = img_hsv[:, :, 2]/255 # matrix of all pixel, normalized
        
        ###############################################
        # compute Michelson contrast
        bri_max = bri.max()
        bri_min = bri.min()
        contrast_Mic = (bri_max - bri_min) / (bri_max + bri_min)
        #print("bri_max - {0}, bri_min - {1}, contrast_Mic -{2}".format(bri_max, bri_min, contrast_Mic))
        #print("The Michelson contrast of this image is {0}".format(contrast_Mic))
        
        # compute dynamic range
        dynamic_range = bri_max - bri_min
        in_features.append(dynamic_range)
        in_features.append(contrast_Mic)
        #print("The dynamic range of this image is {0}".format(dynamic_range))
        
        # compute RMS contrast
        bri_norm = bri / bri_max # 1st, normalize the brightness matrix
        contrast_RMS = bri_norm.std(axis=(0,1))
        in_features.append(contrast_RMS)
#print ("The RMS contrast of this image is {0}".format(contrast_RMS))
    try:
        len_title = len(video_title)
    except Exception as e:
        print("Please enter a video title...")
        in_features.append(len_title)

    return np.array(in_features)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template("input.html")

@app.route('/output', methods=['GET', 'POST'])
def output():
    file = request.files['file']
    
    # add video title
    title = request.form['title']
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    
    return render_template("output.html",
                           filename=file.filename,
                           title=title
                           )

def prediction_advice(filename, title):
    #filename = 'http://127.0.0.1:5000/uploads/' + filename
    filename = "/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/upload/" + filename
    #    filename = "/Users/yaoli/02_JobApplications/03_DataScience/insight/project/websites/web_vblog_diagose/upload/" + filename
    
    #######################
    # input
    #video_title = "DRUGSTORE BACK TO SCHOOL MAKEUP TUTORIAL" # should be input
    video_title = "Philadelphia"
    image_feature = compute_image_features(filename, video_title)
    
    #######################
    # import the model
    clf = joblib.load('model_DTs_global_depth_2.pkl')
    
    #######################
    # predict
    if clf.predict(image_feature.reshape(1, -1)) == 1:
        message = "The thumbnail image and title look AWESOME!!!!"
    else:
        message = "You might want to modify the thumbnail image and title of your video to optimize its impact!"
    return render_template('output.html', message=message)

