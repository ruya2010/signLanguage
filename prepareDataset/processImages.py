import cv2
import numpy as np
import os

# Giriş ve çıkış klasörlerini tanımla
input_folder = "./testMyPhone"
output_folder = "./processed_testMyPhone2"

# Çıkış klasörünü oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_image(image_path):
    # Resmi yükle
    image = cv2.imread(image_path)
    image = cv2.resize(image, (240, 320))

    # Resmi BGR'den RGB'ye dönüştür
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resmi HSV renk uzayına dönüştür
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # cv2.imshow("hsv",hsv_image)

    # Genişletilmiş cilt rengi aralığını belirle
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Maskeleme
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    # cv2.imshow("skin",skin)
    # cv2.imshow("cilt",skin)

    background_color = (255, 255, 255)  # Beyaz renk
    background = np.full_like(image_rgb, background_color)
    # cv2.imshow("bg",background)

    # Cilt rengi maskesi dışındaki bölgeyi arka planla doldur
    non_skin_mask = cv2.bitwise_not(skin_mask)
    # print("non_skin_mask",non_skin_mask.shape)
    # print("background",background.shape)
    background = cv2.bitwise_and(background, background, mask=non_skin_mask)
    result = cv2.bitwise_or(skin, background)
    return result

# res=process_image("./dataset5/s/WhatsApp Image 2024-01-31 at 17.16.08 (1).jpeg")
# cv2.imshow("result",res)
# cv2.waitKey(0)

# Tüm alt klasörlerdeki resimleri işle
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # Resim dosyasının yolu
            image_path = os.path.join(root, file)

            # Resmi işle
            processed_image = process_image(image_path)

            # Yeni dosya adı ve yolunu oluştur
            # folder=os.path.relpath(image_path, input_folder)
            # saveDir=folder.split("/")[0]
            # filename=folder.split("/")[1]
            # output_path = os.path.join(output_folder, saveDir)
            # if (not os.path.exists(output_path)):
                # os.makedirs(output_path)
            output_path=os.path.join(output_folder,file)
            # print(output_path)

            # Çıktıyı kaydet
            cv2.imwrite(output_path,processed_image)

print("Tüm resimler işlendi ve kaydedildi.")
