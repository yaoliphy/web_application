#!/Users/yaoli/anaconda3/bin/python3

import numpy as np
import os
import cv2
import math
import sys
import nltk
import codecs
import pickle
import numpy as np
from numpy import array
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.externals import joblib
from PIL import Image, ImageEnhance
from werkzeug import secure_filename
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from flaskefiles import app


#UPLOAD_FOLDER = '/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/flaskefiles/upload'
UPLOAD_FOLDER = '/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/flaskefiles/static'
STATIC_FOLDER = '/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/flaskefiles/static/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#############################################################################################################
# define functions to get enhance thumbnail image
# compute the input features of an image
# golbal + len_title + date_approx + token (300)
#############################################################################################################


###############################################
# global features
###############################################
def get_image_input_features(img):
    # input: img -> RGB
    img_features_global = []
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #BGR --> HSV
    hue = img_hsv.mean(axis=(0,1))[0]/179
    sat = img_hsv.mean(axis=(0,1))[1]/255
    bri_global = img_hsv.mean(axis=(0,1))[2]/255 # value = brightness
    #print("The global hue, saturation, brightness: ({0}, {1}, {2})".format(hue, sat, bri_global))
    
    bri = img_hsv[:, :, 2]/255 # matrix of all pixel, normalized
    
    # compute Michelson contrast
    bri_max = bri.max()
    bri_min = bri.min()
    contrast_Mic = (bri_max - bri_min) / (bri_max + bri_min)
    #print("bri_max - {0}, bri_min - {1}, contrast_Mic -{2}".format(bri_max, bri_min, contrast_Mic))
    #print("The Michelson contrast of this image is {0}".format(contrast_Mic))
    
    # compute dynamic range
    dynamic_range = bri_max - bri_min
    #print("The dynamic range of this image is {0}".format(dynamic_range))
    
    # compute RMS contrast
    bri_norm = bri / bri_max # 1st, normalize the brightness matrix
    contrast_RMS = bri_norm.std(axis=(0,1))
    #print ("The RMS contrast of this image is {0}".format(contrast_RMS))
    
    img_features_global = [hue, sat, bri_global, dynamic_range, contrast_Mic, contrast_RMS]
    return img_features_global

###############################################
# token indicator
###############################################
def get_token_indicator(video_title, common_words):
    title = nltk.word_tokenize(video_title)
    title = [w.lower() for w in title if w.isalpha()] # the indicator vector for this title
    return [int(x in title) for x in common_words]

###############################################
# all input features
###############################################
def get_all_input_features(img, video_title, common_words):
    # img: RGB
    before_features = get_image_input_features(img)
    before_features.append(len(video_title))
    before_features.append(0)
    before_features = before_features + get_token_indicator(video_title, common_words)
    return (before_features)

###############################################
# obtain enhance images
###############################################
def get_enhanced_image(file_name, video_title, clf, common_words):
    #filename = "/Users/yaoli/02_JobApplications/03_DataScience/insight/project/git_tracked/insight-project/web_app/flaskefiles/upload/" + file_name
    filename = STATIC_FOLDER + file_name
    img_format = Image.open(filename)
    img_original = cv2.imread(filename) #BGR
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) #RGB
    
    features_before = get_all_input_features(img_original, video_title, common_words)
    x_sample = array(features_before).reshape(1, -1)
    prob_before = clf.predict_proba(x_sample)[0, 1] #probability of being popular
    
    if prob_before < 0.5:
        message = "The impact may not be high."
    else:
        message = "The impact is going to be high."
    return message
    
    '''
    #itr = [1, 1.05, 1.1, 1.15]
    itr = [1, 1.05, 1.1, 1.15, 1.2]
    #itr = [0.9, 1, 1.1, 1.2, 1.3
    prob_best = prob_before
    img_best = img_format #RGB, not array, image format
    filter_best = []
    
    for i in itr:
        for j in itr:
            for k in itr:
                #print([i, j, k])
                img_after = ImageEnhance.Color(img_format).enhance(i)
                img_after = ImageEnhance.Brightness(img_after).enhance(j)
                img_after_RGB = ImageEnhance.Contrast(img_after).enhance(k) #RGB
                
                features_after = get_image_input_features(array(img_after_RGB))
                features_after = features_after + features_before[6:]
                x_sample = array(features_after).reshape(1, -1)
                prob_after = clf.predict_proba(x_sample)[0, 1]
                if prob_after > prob_best:
                    img_best = img_after_RGB #RGB
                    prob_best = prob_after
                    filter_best = [i, j, k]
    if prob_best == prob_before:
        message = "The high-impact chance is already very high. So your thumbnail image is kept."
        img_best = img_format
    else:
        message = "The chance of high impact can be improved from {0}% to {1}% by changing saturation by {2}, brightness by {3}, and contrast by {4}".format(100*prob_before, 100*prob_best, filter_best[0], filter_best[1], filter_best[2])

    return (message, img_best)
    '''



#############################################################################################################
# define functions to get enhance thumbnail image
# compute the input features of an image
# golbal + len_title + date_approx + token (300)
#############################################################################################################


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
    print("filename is {0}".format(file.filename))

    #######################
    # import the model
    clf = joblib.load('model_RF_160_estimators_300_tokens_noRegional_noCluster.pkl')

    #######################
    # prepare token list
    with open("title_token_list", 'rb') as f:
        common_words = pickle.load(f)
    common_words = common_words[:300]

    img_name = file.filename
    video_title = title

    message = get_enhanced_image(img_name, video_title, clf, common_words)
    return render_template("output.html", message=message)

    '''
    enhanced_img = get_enhanced_image(img_name, video_title, clf, common_words)
    message = enhanced_img[0]
    img_enchance = enhanced_img[1]
    img_after_name = "enhanced_" + img_name
    img_enchance.save(STATIC_FOLDER + img_after_name, 'JPEG')

    #message = prediction_advice(file.filename, title)

    return render_template("output.html",
                           message=message, name_before=img_name, name_after=img_after_name
                           )
                           '''





