import cv2
from docutils.nodes import classifier
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import ast
import time
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','hastane','i', 'j', 'k', 'l', 'm', 'n', 'o', 'p','polis', 'r', 's','saat', 't', 'u', 'v', 'y', 'z', 'ç', 'ö', 'ü', 'ğ', 'ı', 'ş']  # Toplamda 32 sınıf olduğunu belirtmiştiniz.

with open('./kelimeler/sozluk.txt', 'r') as f:
    lines = f.readlines()

words = [(line.split('. ')[1].strip()).lower() for line in lines]
def complete_word(prefix):
    matches = [word for word in words if word.startswith(prefix)]
    return matches
def find_similar(word):
    max_similarity = 0
    most_similar_words =[]
    for candidate in words:
        similarity = len(set(word) & set(candidate)) / len(set(candidate))
        if similarity > max_similarity or similarity==max_similarity:
            max_similarity = similarity
            most_similar_words.append(candidate)

    return most_similar_words

# Tahmin edilen harfleri saklamak için bir string
sentence = ''
last_letter=""

# Son tahminin zamanını saklamak için bir değişken
last_prediction_time = time.time()
first_prediction=time.time()


print(class_names)
# Load the trained model
model = load_model('./model/18.04/DenseNet121_lr0.0015_1.00.keras')
# model = load_model('./model/19.04/DenseNet121_lr0.0015_1.00.keras')

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# Open the camer
cap = cv2.VideoCapture("/dev/video0")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Check if the camera opened successfully
tahmin_bekle=time.time()
letter_duration=0
while cap.isOpened():
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    # Capture a single frame
    ret, frame = cap.read()
    normal_image=frame.copy()
    frame=cv2.flip(frame,1)
    # Resmi BGR'den RGB'ye dönüştür
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resmi HSV renk uzayına dönüştür
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # cv2.imshow("hsv",hsv_image)
    # Genişletilmiş cilt rengi aralığını belirle
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)

    # Maskeleme
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    # cv2.imshow("cilt",skin)

    background_color = (255, 255, 255)  # Beyaz renk
    background = np.full_like(image_rgb, background_color)

    # Cilt rengi maskesi dışındaki bölgeyi arka planla doldur
    non_skin_mask = cv2.bitwise_not(skin_mask)
    # print("non_skin_mask",non_skin_mask.shape)
    # print("background",background.shape)
    background=cv2.bitwise_and(background,background,mask=non_skin_mask)
    result = cv2.bitwise_or(skin, background)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # cv2.imshow("non_resized_result",result)
    # Geri kalan kısmı griye dönüştür
    # gray_mask = cv2.bitwise_not(skin_mask)
    # gray_background = cv2.cvtColor(cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2GRAY)
    #
    # # Gri arka planı oluştur
    # gray_background_rgb = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)

    # Cilt rengi alanlarını renkli arka plan ile birleştir
    # result = cv2.bitwise_or(skin, background)

    (h, w) = frame.shape[:2]

    # Görüntünün merkezini belirle
    center = (w / 2, h / 2)

    # 90 derece döndürme matrisini oluştur
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Döndürme matrisini ayarla
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Döndürme matrisini uygula
    # frame = cv2.warpAffine(result, M, (nW, nH))
    w=int((w-320)/2)
    h=int((h-240)/2)
    frame=result[w:w+320,h:h+240]
    # frame=255.-frame
    # frame=result
    # cv2.imshow("nonresized",frame)
    # frame = cv2.resize(frame, (400, 800))
    # Check if the frame has been captured successfully
    if not ret:
        print("Can't 0 frame (stream end?). Exiting ...")
        exit()

    # Preprocess the image
    img=frame
    # img = cv2.resize(frame, (240, 320))  # Resize image to match the input size expected by the model
    # img_input = img.astype('float32') / 255  # Normalize pixel values to [0, 1]
    # print(img_input.shape)
    img_input=image.img_to_array(img)
    # img_input= 255. - img_input
    img_input/=255.
    # print(img_input)
    img_input = np.expand_dims(img_input, axis=0) # Add an extra dimension for batch size
    # print(img_input.shape)

    # Use the model to predict the class of the image
    predictions = model.predict(img_input, verbose=0)
    # print(predictions)

    # Get the index of the class with the highest predicted probability
    predicted_class = np.argmax(predictions[0])
    percent = predictions[0][predicted_class]
    cls=class_names.copy()
    print('1. The predicted class name of the image is:', class_names[predicted_class], "percent:", percent)
    # print(predicted_class)
    # print(predictions[0])
    # print(predictions[0][predicted_class])
    # if (class_names[predicted_class]=="b"):
    #     np.delete(predictions[0],predicted_class)
    #     cls.pop(predicted_class)
    #     predicted_class=np.argmax(predictions[0])
    #     percent = predictions[0][predicted_class]
    #     print('2. The predicted class name of the image is:', cls[predicted_class], "percent:", percent)
    # if (cls[predicted_class]=="c"):
    #     np.delete(predictions[0],predicted_class)
    #     cls.pop(predicted_class)
    #     predicted_class=np.argmax(predictions[0])
    #     percent = predictions[0][predicted_class]
    #     print('3. The predicted class name of the image is:', cls[predicted_class], "percent:", percent)
    # if (cls[predicted_class]=="d"):
    #     np.delete(predictions[0],predicted_class)
    #     cls.pop(predicted_class)
    #     predicted_class=np.argmax(predictions[0])
    #     percent = predictions[0][predicted_class]
    #     print('4. The predicted class name of the image is:', cls[predicted_class], "percent:", percent)
    class_name = "???"
    last_letter = class_names[np.argmax(predictions[0])]
    # print(last_letter)
    # Tahminin en yüksek değeri eğer belirli bir eşikten büyükse
    if np.max(predictions[0]) > 0.4 and last_letter == class_names[np.argmax(predictions[0])] and time.time()-tahmin_bekle>2:
        letter_duration+=1
        # En yüksek tahmin değerine sahip olan harfi bul]
        letter = class_names[np.argmax(predictions[0])]
        cv2.putText(img, 'Yuzde:%'+str(predictions[0][predicted_class]*100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, 'Harf:' + letter, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        # Harfi cümleye ekle
        if (time.time()-first_prediction>2.5 and letter_duration>8): #letter_durationkalibre edilecek
            if (len(sentence)>0 and letter!=sentence[-1]):
                sentence += letter
            if (len(sentence)==0):
                sentence += letter
            tahmin_bekle=time.time()
            letter_duration=0

        # Son tahminin zamanını güncelle
        last_prediction_time = time.time()
        if (last_letter!=letter):
            first_prediction=time.time()

    # Eğer son tahminden itibaren belirli bir süre geçtiyse
    elif time.time() - last_prediction_time > 2.0 and (len(sentence)>0 and sentence[-1] != ' '):
        # Cümleye bir boşluk ekle
        sentence += ' '

        # Son tahminin zamanını güncelle
        # last_prediction_time = time.time()

    if (time.time()-last_prediction_time>4.0):
        if sentence[-2:]=="??":
            sentence=sentence[:-3]
        else:
            sentence=sentence[:-2]
    if (time.time()-last_prediction_time>5.0):
        sentence=""

    # Tahmin edilen harfleri ekranın altında göster
    rgb_kare = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb_kare)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 55, encoding="utf-8")
    draw.text((10,400),sentence,fill=(0,255,0),font=font)
    # cv2.putText(normal_image, sentence, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    advices=[]
    if len(sentence)>2:
        kelime=sentence[-2:]
        if (sentence[-1]==" "):
            kelime=sentence[-3:-1]
        matches=complete_word(kelime)
        if len(matches)>0:
            advices=matches
    coor=0
    for advice in advices:
        # cv2.putText(normal_image, advice, (10, coor), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35, encoding="utf-8")
        draw.text((10, coor), advice, fill=(255, 0, 0), font=font)
        coor+=30
    similars = find_similar(sentence)
    coor=0
    for similar in similars:
        # cv2.putText(normal_image, advice, (10, coor), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35, encoding="utf-8")
        draw.text((400, coor), similar, fill=(255, 0, 0), font=font)
        coor+=30
    normal_image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    # if percent > 0.0:
    #     class_name = cls[predicted_class]
    #     cv2.putText(img, 'Yuzde:%'+str(predictions[0][predicted_class]*100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #     cv2.putText(img, 'Harf:' + class_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    #     # print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    # else:
    #     cv2.putText(img, 'Yuzde:%00', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,                   (255, 0, 0), 2)
    #     cv2.putText(img, 'Harf:???', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # print(img.shape)
    # print(normal_image.shape)
    height1, width1, _ = img.shape
    height2, width2, _ = normal_image.shape
    max_height = max(height1, height2)

    # Yeni bir beyaz arka plan oluştur
    final_image = np.ones((max_height, width1 + width2, 3), dtype=np.uint8) * 255

    # İlk resmi beyaz arka plana yerleştir
    final_image[:height1, :width1] = img
    # İkinci resmi beyaz arka plana yerleştir
    final_image[:height2, width1:] = normal_image
    cv2.imshow('sonuc ekrani', final_image)
    # cv2.imshow("normal",normal_image)

    # Release the camera
    if cv2.waitKey(5) == 27:
        break
    pass

cv2.destroyAllWindows()
cap.release()

