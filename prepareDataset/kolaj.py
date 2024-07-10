import os
import shutil
from glob import glob

# klasör yolu
klasor_yolu = './dataset'
hedef_klasor = './kolaj'

# alt klasörler
alt_klasorler = [dI for dI in os.listdir(klasor_yolu) if os.path.isdir(os.path.join(klasor_yolu,dI))]

for klasor in alt_klasorler:
    klasor = os.path.join(klasor_yolu, klasor)
    resimler = glob(klasor + '/*.jpg')  # .jpg resimlerini al, farklı bir format için bunu değiştirin
    if resimler:
        shutil.copy(resimler[0], hedef_klasor)  # ilk resmi hedef klasöre kopyala
