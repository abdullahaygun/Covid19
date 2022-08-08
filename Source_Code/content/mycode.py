import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras import layers
from keras import models
import numpy as np
import tensorflow as tf
DATASET_PATH = 'C:/Users/AYGUN/Desktop/Projelerim/Staj/Projem/content/two/train'
test_dir = 'C:/Users/AYGUN/Desktop/Projelerim/Staj/Projem/content/two/test'
IMAGE_SIZE = (150, 150)
# NUM_CLASSES = len(data_list)
# NUM_CLASSES = 60

# Aynı anda işlenecek olan girdi sayısıdır. Bellek ile doğru orantılıdır. RAM az ise yüksek tutulması önerilmez.
BATCH_SIZE = 10

# Eğitim tur sayısını belirtir.
NUM_EPOCHS = 3

# Öğrenme Hızı yüsek ise daha hızlı öğrenme ancak yanlış öğrenmeye neden olabilir. Hız düşük ise öğrenme daha doğru ancak daha uzun sürecektir.
LEARNING_RATE = 0.0005

# Train datagen here is a preprocessor
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=50, featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.25, zoom_range=0.1, zca_whitening=True, channel_shift_range=20, horizontal_flip=True, vertical_flip=True, validation_split=0.2, fill_mode='constant')
train_batches = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, subset="training", seed=42, class_mode="binary",)
valid_batches = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE, subset="validation", seed=42, class_mode="binary",)

# Öğrenme modeli olarak VGG16 seçildi ve Aktivasyon fonksiyonları olarak relu ve sigmoid kullanıldı.
conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(150, 150, 3))

conv_base.trainable = True
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(
    lr=LEARNING_RATE), metrics=['acc'])

# train_datagen.fit(model)

STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size
STEP_SIZE_VALID = valid_batches.n//valid_batches.batch_size
result = model.fit_generator(train_batches, steps_per_epoch=STEP_SIZE_TRAIN,
                             validation_data=valid_batches, validation_steps=STEP_SIZE_VALID, epochs=NUM_EPOCHS, )

# Plot çizdirmeyi yani grafiği oluşturmaya yarıyor.


def plot_acc_loss(result, epochs):
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(1, epochs), acc[1:], label='Eğitim_Başarı')
    plt.plot(range(1, epochs), val_acc[1:], label='Test_Başarı')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(1, epochs), loss[1:], label='Eğitim_Kayıp')
    plt.plot(range(1, epochs), val_loss[1:], label='Test_Kayıp')
    plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_acc_loss(result, NUM_EPOCHS)

test_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False,
    seed=42,


    class_mode="binary")
eval_generator.reset()
x = model.evaluate_generator(eval_generator, steps=np.ceil(len(
    eval_generator) / BATCH_SIZE), use_multiprocessing=False, verbose=1, workers=1)
print('Test Kaybı: ', x[0])
print('Test Doğruluğu: ', x[1])

eval_generator.reset()
pred = model.predict_generator(eval_generator, 1000, verbose=1)
print("Tahminler Tamamlandı")
for index, probability in enumerate(pred):
    image_path = test_dir + "/" + eval_generator.filenames[index]
    image = mpimg.imread(image_path)
    # BGR TO RGB conversion using CV2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = np.array(image)
plt.imshow(pixels)

print(eval_generator.filenames[index])
if probability > 0.5:
    plt.title("% .2f" % (probability[0]*100) + "% Normal")
else:
    plt.title("% .2f" % ((1-probability[0])*100) + "% COVID19 Pneumonia")
    plt.show()
