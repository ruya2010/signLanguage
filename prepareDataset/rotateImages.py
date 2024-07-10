import os
import cv2

# Ana klasör yolunu belirtin
base_dir = '../dataset'

# Ana klasördeki tüm alt klasörleri ve dosyaları dolaş
for root, dirs, files in os.walk(base_dir):
    # Her dosya için
    s=0
    for file in files:
        # Dosyanın tam yolunu al
        file_path = os.path.join(root, file)
        # Dosyanın bir resim olup olmadığını kontrol et (sadece .jpg ve .png dosyaları)
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            # Resmi oku
            img = cv2.imread(file_path)
            # Resmin genişliği yüksekliğinden büyükse
            print("shape:",img.shape)
            if img.shape[1] > img.shape[0]:
                # Resmi 90 derece döndür
                s+=1
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                # Resmi aynı dosyaya yaz
                cv2.imwrite(file_path, img)
print("şu kadar dosya döndürüldü:",s)