import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import joblib
import glob

st.set_page_config(layout="centered")

model_path = 'resources/fitted_models/'
clf_label = pd.read_csv("resources/clf_label.csv")["0"].to_list()

st.title("준비된 포켓몬 이미지를 넣어서 모델성능을 테스트 해보기")

# st.write(msg)
@st.experimental_singleton
def load_models(model_path):
    clf_model = tf.keras.models.load_model(f'{model_path}model_classification.h5')
    type1_model = tf.keras.models.load_model(f'{model_path}resnet50_type1.h5')
    type2_model = tf.keras.models.load_model(f'{model_path}resnet50_type2.h5')
    ohe = joblib.load("resources/fitted_models/ohe.joblib")
    return clf_model, type1_model, type2_model, ohe

@st.cache
def img_to_4D(img):
    image_resized = cv2.resize(img, (150,150))
    image_array = img_to_array(image_resized)
    image_array = image_array/255.0
    # image_array.shape
    image_array_4D = np.expand_dims(image_array, axis=0)
    return image_array_4D

@st.cache
def read_and_resize_img(path, size = None, show = False):
    img = cv2.cvtColor(cv2.imread(path),
                       cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)
    if show:
        plt.imshow(img)
    return img

clf_model, type1_model, type2_model, ohe = load_models(model_path)



with open('resources/poke_dict.json', 'r') as fp:
   poke_dict = json.load(fp)



test_imgs = glob.glob("resources/img/test_img/*")

test_idx = st.select_slider('테스트 하고 싶은 포켓몬을 골라보세요', range(len(test_imgs)))


test_img = read_and_resize_img(test_imgs[test_idx], size = (300, 300))
st.image(test_img)


start = st.button('예측하기')






if start:
    my_bar = st.progress(0)


    y_pred = clf_model.predict(img_to_4D(test_img))
    y_predict = np.argsort(-y_pred)[:,:3][0]
    my_bar.progress(100)
    pred_type1 = type1_model.predict(img_to_4D(test_img) * 255)
    pred_type2 = type2_model.predict(img_to_4D(test_img) * 255)

    # st.write(pred_type1)
    # st.write(pred_type2)
    
    my_bar.progress(100)
    
    st.subheader("이 포켓몬의 정체는")


    cols = st.columns([1, 1, 1])    
    for col, i in zip(cols, y_predict):
        with col:        
            st.image(f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{poke_dict['idx'][clf_label[i]]}.png",
                    width=150, # Manually Adjust the width of the image as per requirement
                )    
            st.subheader(poke_dict["kor_name"][clf_label[i]])
            st.write("일 확률",format(y_pred[0][i]*100, '.2f'),"% 입니다.")

    type1_predict = np.argsort(-pred_type1, axis = 1)[:,: 3][0]
    type2_predict = np.argsort(-pred_type2, axis = 1)[:,: 3][0]

    st.subheader("")
    st.subheader("")


    st.subheader("이 포켓몬의 타입은")

    cols = st.columns([1, 1, 1])  

    for col, t1, t2 in zip(cols, type1_predict, type2_predict):
        with col:
            st.write('타입 1: ', ohe.categories_[0][t1])
            st.write('타입 2: ', ohe.categories_[0][t2])
