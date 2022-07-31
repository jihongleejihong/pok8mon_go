import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import json
import joblib
import glob
from PIL import Image
import multiprocessing
import requests


@st.experimental_singleton 
def load_models(model_path):
    clf_model = tf.lite.Interpreter(f'{model_path}model_classification.tflite', num_threads = multiprocessing.cpu_count())
    clf_model.allocate_tensors()
    type1_model = tf.lite.Interpreter(f'{model_path}resnet50_type1.tflite', num_threads = multiprocessing.cpu_count())
    type1_model.allocate_tensors()
    type2_model = tf.lite.Interpreter(f'{model_path}resnet50_type2.tflite', num_threads = multiprocessing.cpu_count())
    type2_model.allocate_tensors()
    ohe = joblib.load("resources/fitted_models/ohe.joblib")
    return clf_model, type1_model, type2_model, ohe


def img_to_4D(img):
    image_resized = cv2.resize(img, (150,150))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = image_array
    # image_array.shape
    image_array_4D = np.expand_dims(image_array, axis=0)
    return image_array_4D


def read_and_resize_img(path, size = None, show = False):
    img = cv2.cvtColor(cv2.imread(path),
                    cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)
    if show:
        plt.imshow(img)
    return img


def model_predict(interpreter, test_img):
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], test_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data


def upper_type(type_label):
    return type_label[0].upper() + type_label[1:]


@st.cache
def load_dict():
    with open('resources/poke_dict.json', 'r') as fp:
        poke_dict = json.load(fp)

    with open('resources/kor_type.json', 'r') as fp:
        type_dict = json.load(fp)
    return poke_dict, type_dict


def main():
    with tab1:
        st.markdown('''
        # Pokémon Prediction by Images

        ## 🐭 Pok8mon Go ! ⚡️
        #### 참여자: 김현욱, 박이슬, 이명진, 이지홍''')
        st.markdown('---')
    
        st.image('resources/img/main_img/0.jpeg',
        width = 700)
    with tab2:
        st.markdown('''
        ### 프로젝트 목적
        
        - 합성곱신경망(Convolutional Neural Network, CNN)을 이용한 이미지 처리에 대한 이해
        - 인공지능 서비스 개발 과정에 대한 이해 및 수행

        ''')
        
        st.image('resources/img/main_img/1.png', width = 800)

        st.markdown('---')

        st.markdown('''
        ### 프로젝트 Workflows
        ''')

        st.image('resources/img/main_img/2.png', width = 800)

        st.markdown('''
        - **데이터 수집 및 가공**
            - Pokemon `Image dataset` (이하, 이미지 데이터)
            - Pokemon `기본 정보 (Stats. 포함) dataset` (이하, Stats. 데이터)
            
        - **모델 학습**
            
            모델 1, 2는 각 문제에 적절한 pre-trained model을 기반으로 학습한 반면, 
            
            모델 3 학습에서는 전이 학습을 이용하지 않았음
            
            - 모델 1: 이미지 데이터 학습을 통한 Pokemon **종류 분류**
            - 모델 2: 이미지 데이터 학습을 통한 Pokemon **속성(Type) 분류** (ex. 피카츄 → Electric)
            - 모델 3: Stats. 데이터 학습을 통한 Pokemon **Stats. 예측**
            
        - **API 배포 (이 부분 최종 배포 후 수정 필요)**
            - 데이터 EDA
            - 모델(모델 1, 모델 2) 학습에 따른 Pokemon 종류 / 속성 Classification 결과
            - 사용자 Input(이미지 파일)과 가장 유사한 Pokemon 종류 분류
        ''')

        st.markdown('---')

        st.markdown('''

        ### 프로젝트 요약
            
        - 본 프로젝트를 수행하며 포켓몬 149종에 관한 데이터를 활용하였음. 
            
        -  6825개 이미지와 800개 Stats. 데이터를 과제 별 목적에 맞게 가공하여 모델 학습에 이용함.

        ''')

        st.markdown('''
        | 과제 | Dataset 원본 | Train set | Test set | source |
        | --- | --- | --- | --- | --- |
        | 1. 종류 분류 | (6784, 150, 150, 3) | (5427, 150, 150, 3) | (1357, 150, 150, 3) | [이미지] |
        | 2. 속성 분류 | (6825, 150, 150, 3) | (5324, 150, 150, 3) | (1501, 150, 150, 3) | [능력치] |
        | 3. Stats. 예측 | (800, 13) |  |  | [능력치] |

        > 데이터 원본
        > * Image dataset - [[Kaggle - 7000 hand-cropped and labeled Pokemon images for classification]](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) 
        > * Stats. dataset - [[Kaggle - 721 Pokemon with stats and types](https://www.kaggle.com/datasets/abcsds/pokemon)]
        ''')

        st.markdown('---')

        st.markdown('''
        ### 데이터 미리 보기

        - 이미지 데이터

        아래와 같이 각 포켓몬에 대해 다양한 각도, 배경, 효과가 적용된 여러 이미지를 학습에 적용함 

        (포켓몬 149종 6825개 → 1마리 당 평균 46개 이미지 학습)

        ''')

        cols = st.columns(2)
        with cols[0]:
            st.image('resources/img/main_img/3.png', width = 400)
        with cols[1]:
            st.image('resources/img/main_img/4.png', width = 400)

        st.markdown("- 스탯 데이터")

        st.image('resources/img/main_img/5.png', width = 800)

        cols = st.columns(2)
        with cols[0]:
            with st.expander('[데이터 컬럼 정보]'):
                st.markdown('''
            - `#, Name`: 포켓몬 별 고유 번호, 이름
            - **`Type 1, Type 2`**: 포켓몬 별 고유 속성 ⇒ 특정 공격에 대해 취약 / 방어력이 높음을 의미



            - `Total` : 모든 Stats. 정보(HP~Speed)의 합 ⇒ 해당 포켓몬이 얼마나 강한지를 의미
                - **`HP`**: 포켓몬이 견딜 수 있는 데미지의 총합
                - **`Attack`**: Scratch, Punch 등 일반 공격의 타격
                - **`Defense`**: 일반 공격에 대한 데미지 저항
                - **`Sp. Atk`**: Fire blast, Bubble beam 등 특정 포켓몬에 특화된 공격의 타격
                - **`Sp. Def`**: Special attack에 대한 데미지 저항
                - **`Speed`** : 포켓몬의 속도 ⇒ 선제 공격 여부 결정
            - `Generation`: 포켓몬 세대 (세대가 진화할 수록, 일부 Type이 추가됨)
            - `Legendary`: 전설의 희귀 포켓몬 여부 (bool)

                ''')

        st.markdown('---')    
        st.markdown('### 프로젝트 결과')

        df_acc = pd.DataFrame([['Name', 'Train', 0.992],
        ['Name', 'Test', 0.893],
        ['Type1', 'Train', 0.909],
        ['Type1', 'Test', 0.879],
        ['Type2', 'Train', 0.938],
        ['Type2', 'Test', 0.917],], columns = ['Target', 'Condition', 'Accuracy'])
        ct_acc = pd.crosstab(columns = df_acc["Target"], index = df_acc["Condition"], values = df_acc["Accuracy"], aggfunc= np.sum).sort_index(ascending = False)

        st.markdown('#### **포켓몬 종류 분류 (Name)**')
        st.image('resources/img/main_img/6.png', width = 800)
        st.markdown("`Accuracy`")
        st.dataframe(ct_acc["Name"])



        st.markdown(
        '#### **포켓몬 속성 분류 (Type 1, Type2)**')
        st.image('resources/img/main_img/7.png', width = 800)
        st.markdown("`Accuracy`")
        st.dataframe(ct_acc[["Type1", "Type2"]])
    
    with tab3:
        model_path = 'resources/fitted_models/'
        clf_label = pd.read_csv("resources/clf_label.csv")["0"].to_list()

        clf_model, type1_model, type2_model, ohe = load_models(model_path)

        poke_dict, type_dict = load_dict()



        title_options = ["준비된 이미지로 모델 테스트", "당신은 어떤 포켓몬인가요?"]
        clf_title_options = ["이 포켓몬의 정체는", "당신의 정체는 바로"]
        type_title_options = ["이 포켓몬의 타입은", "당신의 타입은"]

        task = st.selectbox('선택:', title_options)
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
            method = st.selectbox('이미지 입력 방식을 골라주세요', ['파일 업로드', '이미지 링크로 불러오기', '카메라로 찍기'])
            user_img = None
            if method == '파일 업로드':
                user_img = st.file_uploader('이미지를 업로드하세요', type = ['png', 'jpg', 'jpeg'])
            elif method == '이미지 링크로 불러오기':
                url = ''
                url = st.text_input('이미지 주소를 입력 후 Enter')
                if url:
                    user_img = requests.get(url, stream=True).raw
            else:    
                user_img = st.camera_input("사진을 찍어주세요")

            if user_img:
                test_img = np.asarray(Image.open(user_img).convert('RGB').resize((300, 300)))
                if not method == '카메라로 찍기':
                    st.image(test_img)
            


        if test_img is not None:
            start = st.button('예측하기')


            st.subheader("")
            st.subheader("")


            if start:
                my_bar = st.progress(0)

                col1, col2 = st.columns(2)
                with col1:
                    prediction_state = st.text('포켓몬 이름 분류중')
                    y_pred = model_predict(clf_model, img_to_4D(test_img/255.0))
                    y_predict = np.argsort(-y_pred)[:,:3][0]
                    prediction_state.success('포켓몬 이름 분류 완료')
                my_bar.progress(33)
                with col2:
                    prediction_state = st.text('포켓몬 타입 분류 중')
                    pred_type1 = model_predict(type1_model, np.uint8(img_to_4D(test_img)))
                    my_bar.progress(66)
                    pred_type2 = model_predict(type2_model, np.uint8(img_to_4D(test_img)))        
                    my_bar.progress(100)
                    prediction_state.success('포켓몬 타입 분류 완료')
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



st.set_page_config(layout="centered", page_title="Pokémon Prediction by Images", page_icon = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQia9FUVgdDp2rNa6SGUobh7ywxR9UqEyuJpAMjUulfyxt-ls7aEhh6uLSvNJ-_bFCZs5w&usqp=CAU',
    )

tab1, tab2, tab3 = st.tabs(['Home', 'About', 'Play with models']) 

main()
