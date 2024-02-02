from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Učitavanje spremljenog modela
model = load_model('model.h5')

# Učitavanje testne slike
test_image = load_img('S2_kontura18_11.png', target_size=(40, 40))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)#dodaje dotatnu dimenziju koja oznacava broj primera u ovom slucaju 1 kako bu se dobio rezultat za sliku koja se prosledi


# Predviđanje na testnoj slici
result = model.predict(test_image)
prediction = np.argmax(result)#Vrenosti 0 / 1 / 2
confidence = np.max(result)

if prediction == 0:
    prediction_label = 'popunjen'
elif prediction == 1:
    prediction_label = 'prazan'
else:
    prediction_label = 'neispravno popunjen'

print(prediction)
print(confidence)
print(prediction_label)
