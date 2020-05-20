from PIL import Image
import numpy as np
import os
import json
import requests


def crop2img2arr(path, in_shape=(128, 128), crop=False, center=False):
    try:
        img = Image.open(path, 'r')
        if crop:
            if img.size[0] >= img.size[1]:
                img = img.crop((0, 0, img.size[0], img.size[0]))
            else:
                img = img.crop((0, 0, img.size[1], img.size[1]))
        img = img.resize(in_shape)
        img = img.convert('RGB')
        if center:
            imgarr = np.array(img, dtype='float32')
            imgarr = imgarr - imgarr.mean()
            imgarr = imgarr / max(imgarr.max(), abs(imgarr.min()))
        else:
            imgarr = np.array(img, dtype='float32')
            imgarr /= 128.
            imgarr -= 1.
        return imgarr
    except:
        return None


def save2json(arr, name='post.json'):
    arr = np.expand_dims(arr, axis=0)
    with open(name, 'w') as file_object:
        json.dump({"signature_name": "serving_default",
                   "instances": arr.tolist()}, file_object)


def compare_imgs(names, main_dir):
    arr = []
    for name in names:
        path2img = os.path.join(main_dir, name)
        img2arr = crop2img2arr(path2img)
        if img2arr is not None:
            arr.append(img2arr)
    return np.array(arr)


def main():
    url = 'http://127.0.0.1:5000/predict'
    # headers = {'Content-type': 'application/json',
    #            'Accept': 'text/plain',
    #            'Content-Encoding': 'utf-8'}

    # files = {'img': open(r'C:\Users\vladislav.sabenin\PycharmProjects\speech-driven-animation\example\image.bmp', 'rb'),
    #          'audio': open(r'C:\Users\vladislav.sabenin\PycharmProjects\speech-driven-animation\example\audio.wav', 'rb')}

    files = {'img': open(r'D:\Data_Img\_TESTNG\3\08_w_2b70ecac.jpg', 'rb'),
             'audio': open(r'D:\Mtorrent\Music_Child\111. Бармалей.mp3', 'rb')}
    answer = requests.post(url, files=files)

    print(answer.status_code)
    with open(r'temp\new_video.mp4', 'wb') as f:
        f.write(answer.content)


if __name__ == '__main__':
    main()
