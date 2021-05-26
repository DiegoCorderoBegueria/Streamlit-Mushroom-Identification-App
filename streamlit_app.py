import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
import numpy as np
import pandas as pd
from './NDScaler' import NDScaler


labels = pd.read_csv(
    'https://raw.githubusercontent.com/DiegoCorderoBegueria/Streamlit_MushroomCM/main/probability-labels-en.txt',
    delimiter="\t",
    header=None)

model = hub.load("https://tfhub.dev/bohemian-visual-recognition-alliance/models/mushroom-identification_v1/2")

st.header('Mushroom identification')
file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])

scaler = NDScaler()


def preprocessing_img(image):
    img = Image.open(image)
    test_image = img.resize((360, 360))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    scaler.fit(test_image)
    test_image = scaler.transform(test_image)
    test_image = [test_image]
    return test_image


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 360, 360, 3], dtype=tf.float32)])
def call_model(image):
    output_dict = model.signatures["serving_default"](image)
    return output_dict['out']


if file_uploaded is not None:
    fig, ax = plt.subplots()
    file = Image.open(file_uploaded)
    plt.imshow(file)
    plt.axis("off")
    file = preprocessing_img(file_uploaded)
    predictions = call_model(file)
    label_predicted = labels[0].iloc[np.argmax(predictions)]
    proba_predicted = (100 * np.max(predictions)).round(2)
    result = f"This mushroom is a {label_predicted} with a {proba_predicted}% probability."
    st.write(result)
    st.pyplot(fig)
