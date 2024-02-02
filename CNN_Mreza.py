from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
#Kreiranje arhitekture mreze
classifier = Sequential()
classifier.add(Conv2D(64, (3, 3), input_shape=(40, 40, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)#Priprema podataka za trning i validaciju
val_datagen = ImageDataGenerator(rescale=1./255)#Pikseli u slikama se skaliraju na opseg od 0 do 1

train_generator = train_datagen.flow_from_directory(
    'Trening',
    target_size=(40, 40),
    batch_size=25,#broj slika koji se prolsedjuje u jendom prolazu
    class_mode='categorical',#definisanje viseklasne klasifikacije
    classes=['popunjen', 'nepopunjen', 'neispravno_popunjen']
)

val_generator = val_datagen.flow_from_directory(
    'TreningV',
    target_size=(40, 40),
    batch_size=25,
    class_mode='categorical',
    classes=['popunjen', 'nepopunjen', 'neispravno_popunjen']
)

classifier.fit(
    train_generator,                      #Tok podataka za trening generisan pomoću train_generator objekta.
    steps_per_epoch=29,                   #Broj koraka (batches) koji će se izvršiti u jednoj epohi. 
    epochs=8,                             #Broj epoha, odnosno koliko puta će se model trenirati na celom trening setu.
    validation_data=val_generator,        #Tok podataka za validaciju generisan pomoću val_generator objekta.
    validation_steps=22                   #Broj koraka (batches) koji će se izvršiti prilikom validacije u svakoj epohi. 
)

classifier.save('modelV2.h5')

test_image = load_img('roiT1_2.png', target_size=(40, 40))
test_image = np.array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
prediction = np.argmax(result)
confidence = np.max(result)

if prediction == 0:
    prediction_label = 'popunjen'
elif prediction == 1:
    prediction_label = 'prazan'
else:
    prediction_label = 'neispravno popunjen'

print(confidence)
print(prediction_label)
