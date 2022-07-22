import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json

model_path = 'resources/fitted_models/'
clf_label = ['Weezing', 'Magneton', 'Horsea', 'Rhydon', 'Meowth', 'Weedle', 'Machoke', 'Ivysaur', 'Vulpix', 'Snorlax', 'Persian', 'Rhyhorn', 'Cloyster', 'Mew', 'Poliwrath', 'Muk', 'Wartortle', 'Haunter', 'Koffing', 'Drowzee', 'Clefairy', 'Golem', 'Grimer', 'Venomoth', 'Venonat', 'Gastly', 'Gyarados', 'Omastar', 'Sandslash', 'Pidgey', 'Lickitung', 'Geodude', 'Raticate', 'Oddish', 'Dragonair', 'Hypno', 'Pidgeot', 'Charmeleon', 'Beedrill', 'Rapidash', 'Nidorina', 'Shellder', 'Jynx', 'Kingler', 'Poliwhirl', 'Arcanine', 'Diglett', 'Magikarp', 'Bellsprout', 'Eevee', 'Slowpoke', 'Caterpie', 'Golbat', 'Alolan Sandslash', 'Aerodactyl', 'Growlithe', 'Spearow', 'Venusaur', 'Psyduck', 'Gengar', 'Kadabra', 'Dragonite', 'Sandshrew', 'Farfetchd', 'Golduck', 'Metapod', 'Marowak', 'Kakuna', 'Arbok', 'Ditto', 'Gloom', 'Blastoise', 'Krabby', 'Jigglypuff', 'Machamp', 'Seadra', 'Hitmonchan', 'Porygon', 'Onix', 'Electrode', 'Pikachu', 'Alakazam', 'Vaporeon', 'Hitmonlee', 'Charizard', 'Mankey', 'Doduo', 'Fearow', 'Zubat', 'Staryu', 'Butterfree', 'Slowbro', 'Victreebel', 'Squirtle', 'Pidgeotto', 'Tentacool', 'Parasect', 'Bulbasaur', 'Omanyte', 'Clefable', 'Moltres', 'Zapdos', 'Primeape', 'Kabuto', 'Scyther', 'Nidoqueen', 'Dratini', 'Electabuzz', 'Nidorino', 'Exeggcute', 'Chansey', 'Pinsir', 'Tentacruel', 'Paras', 'Graveler', 'Vileplume', 'Articuno', 'Ponyta', 'Rattata', 'Magnemite', 'Flareon', 'Tangela', 'Mewtwo', 'Cubone', 'Starmie', 'Weepinbell', 'Lapras', 'Raichu', 'Seel', 'Dodrio', 'Dugtrio', 'Magmar', 'Kabutops', 'MrMime', 'Seaking', 'Voltorb', 'Tauros', 'Abra', 'Nidoking', 'Poliwag', 'Charmander', 'Ekans', 'Jolteon', 'Ninetales', 'Wigglytuff', 'Dewgong', 'Machop', 'Kangaskhan', 'Goldeen', 'Exeggutor']


st.title("준비된 포켓몬 이미지를 넣어서 모델성능을 테스트 해보기")



# msg = "Now loading models"

# st.write(msg)
clf_model = tf.keras.models.load_model(f'{model_path}model_classification.h5')
# type1_model = tf.keras.models.load_model(f'{model_path}resnet50_type1.h5')
# type2_model = tf.keras.models.load_model(f'{model_path}resnet50_type2.h5')

# msg = ""

with open('resources/poke_dict.json', 'r') as fp:
   poke_dict = json.load(fp)

def read_and_resize_img(path, size = None, show = False):
    img = cv2.cvtColor(cv2.imread(path),
                       cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)
    if show:
        plt.imshow(img)
    return img


test_img = read_and_resize_img('resources/img/test_img/IMG_9536.jpeg', size = (300, 300))


st.write('테스트 이미지')
st.image(test_img)

def img_to_4D(img):
    image_resized = cv2.resize(img, (150,150))
    image_array = img_to_array(image_resized)
    image_array = image_array/255.0
    image_array.shape
    image_array_4D = np.expand_dims(image_array, axis=0)
    return image_array_4D

y_pred = clf_model.predict(img_to_4D(test_img))

y_predict = np.argsort(-y_pred)[:,:5]

from PIL import Image
for i in y_predict[0]:
    st.image(f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{poke_dict['idx'][clf_label[i]]}.png",
            width=150, # Manually Adjust the width of the image as per requirement
        )    
    st.write(poke_dict["kor_name"][clf_label[i]], end=' ')
    st.write("일 확률",format(y_pred[0][i]*100, '.2f'),"% 입니다.")