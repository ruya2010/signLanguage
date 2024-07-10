import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Modeli yükleyin
model = load_model('./model/17.04/DenseNet121_lr0.0012_1.00.keras')

# Tahmin yapılacak resimlerin bulunduğu klasör
image_dir = 'testImages/testMyPhone'

# Sınıf isimlerinizi buraya koyun
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','hastane','i', 'j', 'k', 'l', 'm', 'n', 'o', 'p','polis', 'r', 's','saat', 't', 'u', 'v', 'y', 'z', 'ç', 'ö', 'ü', 'ğ', 'ı', 'ş']  # Toplamda 33 sınıf olduğunu belirtmiştiniz.
print(len(class_names))
# Resimlerin bulunduğu klasördeki her dosya için döngü
s=0
for file_name in os.listdir(image_dir):
    # Resmi yükleyin ve yeniden boyutlandırın
    img = cv2.imread(os.path.join(image_dir, file_name))
    img = cv2.resize(img,(240,320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    cv2.imwrite("./tahminResim/16.04.jpg",img)
    # img = image.load_img(os.path.join(image_dir, file_name), target_size=(320, 240))
    # print(img)
    # break
    # img.save("./tahminResim/test12.04.jpg")
    # s+=1
    # img.save("./tahminResim/"+str(s)+".jpg")

    # Resmi bir numpy dizisine dönüştürün
    x = image.img_to_array(img)
    x /= 255.
    # print(x)
    # print(x.shape)
    # x=255.-x
    # Resmi bir batch'e dönüştürün (Resimlerin toplu işlem boyutunu ekleyin)
    x = np.expand_dims(x, axis=0)

    # Veriyi ön işleme yapın (örneğin, [0, 1] aralığına ölçeklendirme)
    # print(x)
    # break
    # print(x.shape)
    # Resim için tahminler yapın
    preds = model.predict(x,verbose=0)

    # En yüksek olasılıklı tahminin indeksini alın
    top_pred = np.argmax(preds[0])
    if (preds[0][top_pred] * 100>0):
        # En yüksek olasılıklı tahminin sınıf adını ve yüzdesini yazdırın
        print(f"Resim: {file_name} - Tahmin: {class_names[top_pred]} - Olasılık: {preds[0][top_pred] * 100}%")
