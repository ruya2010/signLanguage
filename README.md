# İşitme Engelliler için İşaret Dili Tanıma ve Metin Dönüştürme

Bu proje, işitme engelliler için işaret dili harflerini tanıyarak bu harfleri yazıya dönüştüren bir yapay zeka modeli üzerinde çalışmaktadır.

## Proje Yapısı

- **datasets/**: Kullanılan veri setlerini içeren klasör.
- **train/**: Eğitim için hazırlanan dosyaları içeren klasör.
- **test/**: Test için hazırlanan dosyaları içeren klasör.
- **src/**: Projenin Python kodlarını içeren klasör.

## Kullanılan Teknolojiler

- Python
- TensorFlow
- OpenCV
- pyCharm IDE

## Kurulum

1. Repoyu yerel makinenize klonlayın:

    ```bash
    git clone https://github.com/ruya2010/signLanguage.git
    ```

2. Python 3.x'i yükleyin ve gerekli kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

1. Eğitim veri setlerini kullanarak modeli eğitmek için:

    ```bash
    python AiTrainScript/TrainDenseNet.py
    ```

2. Test veri setlerini kullanarak modeli değerlendirmek için:

    ```bash
    python /testCam.py
    ```
## Katkıda Bulunma

1. Repoyu fork edin.
2. Yeni bir branch oluşturun: `git checkout -b yeni-özellik`
3. Yaptığınız değişiklikleri commit edin: `git commit -am 'Yeni özellik eklendi'`
4. Değişikliklerinizi forked repoya push edin: `git push origin yeni-özellik`
5. Bir pull request oluşturun.

## Lisans

Bu proje GPLv3.0 Lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakınız.




