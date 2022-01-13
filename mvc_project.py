from audioop import mul
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

x = np.load('image.npz')['arr_0']
y = pd.read_csv('alphabets_labels.csv')['labels']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 9, train_size = 3500, test_size = 500)

xtrain_scale = xtrain / 255.0
xtest_scale = xtest / 255.0

log = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xtrain_scale, ytrain)

def get_prediction(image):
    im_pil = Image.open(image)

    image_bw = im_pil.convert('L')

    image_resize = image_bw.resize((28, 28), Image.ANTIALIAS)

    pixelfilter = 20

    minpix = np.percentile(image_resize, pixelfilter)

    image_inverted = np.clip(image_resize - minpix, 0, 255)

    maxpix = np.max(image_resize)

    image_inverted = np.asarray(image_inverted) / maxpix

    test_sample = np.array(image_inverted).reshape(1, 660)

    test_prediction = log.predict(test_sample)

    return test_prediction[0]
