import requests
from io import BytesIO
from PIL import Image

def img_loader():
    # 다운받을 이미지 url
    url = f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{poke_dict[labels[0]]}.png"

    # request.get 요청
    res = requests.get(url)

    #Img open

    request_get_img = Image.open(BytesIO(res.content))
    print(labels[0])
    return request_get_img