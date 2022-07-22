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
from PIL import Image

st.set_page_config(layout="centered")

model_path = 'resources/fitted_models/'
clf_label = pd.read_csv("resources/clf_label.csv")["0"].to_list()
target_stat = pd.read_csv("resources/target_stats.csv")


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

def upper_type(type_label):
    return type_label[0].upper() + type_label[1:]

clf_model, type1_model, type2_model, ohe = load_models(model_path)

with open('resources/poke_dict.json', 'r') as fp:
   poke_dict = json.load(fp)

with open('resources/kor_type.json', 'r') as fp:
   type_dict = json.load(fp)



title_options = ["준비된 이미지로 모델 테스트", "당신이 포켓몬이라면?"]
clf_title_options = ["이 포켓몬의 정체는", "당신의 정체는 바로"]
type_title_options = ["이 포켓몬의 타입은", "당신의 타입은"]

task = st.sidebar.radio('선택:', title_options)
task_idx = title_options.index(task)
st.title(task)
test_img = None
if task_idx == 0:
    test_imgs = glob.glob("resources/img/test_img/*")
    test_img_dict = {test_img.split('/')[-1].split('.')[0] : idx for idx, test_img in enumerate(test_imgs)}
    test_img_key = st.selectbox('테스트 하고 싶은 포켓몬을 골라보세요', test_img_dict)
    
    col1, col2 = st.columns(2)
    # with col1:        
    test_img = read_and_resize_img(test_imgs[test_img_dict[test_img_key]], size = (300, 300))        
    st.image(test_img)

    # with col2: 
        # target_info = target_stat.loc[target_stat["Korname"] == '뿔카노']
        # actual_name, actual_type1, actual_type2 = target_info[["Korname", "Type 1", "Type 2"]].values[0]
        # st.subheader('예측할 정보')
        # st.write(f"이름: {actual_name}")
        # st.image(f'resources/img/type/{actual_type1}.png')
        # st.write(f"타입 1: {type_dict[actual_type1]}")
        # st.image(f'resources/img/type/{actual_type2}.png')
        # st.write(f"타입 2: {type_dict[actual_type2]}")


else:    
    method = st.selectbox('Pick one', ['파일 업로드', 'PC 카메라로 찍기'])
    if method == '파일 업로드':
        user_img = st.file_uploader('이미지를 업로드하세요', type = ['png', 'jpg', 'jpeg'])
    else:    
        user_img = st.camera_input("사진을 찍어주세요")

    if user_img:
        test_img = np.asarray(Image.open(user_img).convert('RGB').resize((300, 300)))
        if method == '파일 업로드':
            st.image(test_img)
  

if test_img is not None:
    start = st.button('예측하기')


    st.subheader("")
    st.subheader("")


    if start:
        my_bar = st.progress(0)


        y_pred = clf_model.predict(img_to_4D(test_img))
        y_predict = np.argsort(-y_pred)[:,:3][0]
        my_bar.progress(100)
        pred_type1 = type1_model.predict(img_to_4D(test_img) * 255)
        pred_type2 = type2_model.predict(img_to_4D(test_img) * 255)
        
        my_bar.progress(100)
        
        st.subheader(clf_title_options[task_idx])
        

        cols = st.columns([1, 1, 1])    
        for col, i in zip(cols, y_predict):
            with col:        
                st.image(f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{poke_dict['idx'][clf_label[i]]}.png",
                        width=150, # Manually Adjust the width of the image as per requirement
                    )    
                st.subheader(poke_dict["kor_name"][clf_label[i]])
                st.write('확률: ',format(y_pred[0][i]*100, '.2f'),"%")

        type1_predict = np.argsort(-pred_type1, axis = 1)[:,: 3][0]
        type2_predict = np.argsort(-pred_type2, axis = 1)[:,: 3][0]

        st.subheader("")
        st.subheader("")


        st.subheader(type_title_options[task_idx])


        cols = st.columns([1, 1, 1, 1])  
        with cols[1]:
            st.subheader('타입 1')
        with cols[2]:
            st.subheader('타입 2')

        for  t1, t2 in zip(type1_predict, type2_predict):
            type1 = upper_type(ohe.categories_[0][t1])
            type2 = upper_type(ohe.categories_[0][t2])
            with cols[1]:            
                st.image(f'resources/img/type/{type1}.png')
                st.subheader(type_dict[type1])
                st.write('확률: ',format(pred_type1[0][t1]*100, '.2f'),"%")
            with cols[2]:                       
                st.image(f'resources/img/type/{type2}.png')
                st.subheader(type_dict[type2])
                st.write('확률: ',format(pred_type2[0][t2]*100, '.2f'),"%")