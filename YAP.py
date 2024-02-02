import cv2
import numpy as np
from imutils import contours
from collections import OrderedDict
import os
import math
import json
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import ttk
import mysql.connector
from PIL import Image

#==========================================================================================================================================================================================#
def obrisi_slike():
    folder = '.'  # Postavite putanju do foldera u kojem se nalaze slike
    for file_name in os.listdir(folder):
        if file_name.startswith('roiT78') and file_name.endswith('.png'):
            file_path = os.path.join(folder, file_name)
            os.remove(file_path)           
#==========================================================================================================================================================================#
def obrisi_json_fajlove():
    folder = '.'  # Postavite putanju do foldera u kojem se nalaze JSON fajlovi
    for file_name in os.listdir(folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder, file_name)
            os.remove(file_path)
#=======================================================================FUNKCIJA PROMENE DIMEZNZIJE SLIKA==================================================================================#
def resize_images():
    input_paths = filedialog.askopenfilenames(title="Izaberite slike za promenu dimenzija", filetypes=(("Image files", "*.png;*.jpg"), ("All files", "*.*")))
    
    if not input_paths:
        return  # Korisnik nije izabrao slike
    
    for input_path in input_paths:
        # Učitavanje slike
        image = Image.open(input_path)
         # Konvertovanje slike u režim RGB (ako nije već u tom režimu)
        if image.mode != "RGB":
            image = image.convert("RGB")
                
        # Promena dimenzija slike na 715x1010
        resized_image = image.resize((715, 1010))
        
        # Čuvanje izlazne slike na isto mesto kao i originalna slika
        resized_image.save(input_path, format="JPEG", quality=100, optimize=True)
        
    #     print(f"Promenjene dimenzije slike: {input_path}")
    
    # print("Završeno!")
#=========================================================================================================================================================================================#
#---------------------------------------------------------Ucitavanje i rotiranje slike SABLONA---------------------------------------------------------------#
img0 = cv2.imread('Sablon.png')

# Dobijanje dimenzija slike
height0, width0, _ = img0.shape

# Konvertovanje u grayscale
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# Binarizacija
_0, thresh0 = cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Pronalazak kontura
contours0, hierarchy0 = cv2.findContours(thresh0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Pronalazak najveće konture
max_contour0 = max(contours0, key=cv2.contourArea)

# Aproksimacija konture
epsilon0 = 0.1 * cv2.arcLength(max_contour0, True)
approx0 = cv2.approxPolyDP(max_contour0, epsilon0, True)

# Definisanje boja tačaka u odgovarajućem redosledu
colors0 = OrderedDict([('zelena', (0, 255, 0)),
                      ('plava', (255, 0, 0)),
                      ('zuta', (0, 255, 255)),
                      ('braon', (42, 42, 165))])

# Obeležavanje tačaka sa odgovarajućom bojom
for i, point in enumerate(approx0):
    # Provera položaja tačke na slici
    if point[0][1] > height0 / 2:  # Donji deo slike
        if i % 2 == 0:
            color_name0 = 'plava'
        else:
            color_name0 = 'zuta'
    else:  # Gornji deo slike
        if i % 2 == 0:
            color_name0 = 'braon'
        else:
            color_name0 = 'zelena'

    color0 = colors0[color_name0]

    #Ispisivanje vrednosti i značenja tačaka
        #print("Koordinate tačke {}: {}".format(i+1, tuple(point[0])))
        #print("Boja tačke {}: {}".format(i+1, color_name0))

n0 = len(approx0)
slopes0 = []  # Prazna lista za čuvanje nagiba

bottom_left_index0 = [i for i, point in enumerate(approx0) if point[0][0] < width0 / 2 and point[0][1] > height0 / 2][0]
bottom_left_point0 = approx0[bottom_left_index0][0]

#cv2.circle(img0, tuple(bottom_left_point0), 5, (0, 0, 255), -1)
#print("Koordinate donje leve tačke: ({}, {})".format(bottom_left_point0[0], bottom_left_point0[1]))

bottom_right_index0 = [i for i, point in enumerate(approx0) if point[0][0] > width0 / 2 and point[0][1] > height0 / 2][0]
bottom_right_point0 = approx0[bottom_right_index0][0]

diff_x0 = bottom_right_point0[0] - bottom_left_point0[0]
diff_y0 = bottom_right_point0[1] - bottom_left_point0[1]
rotation_angle_rad0 = math.atan(diff_y0 / diff_x0)

if diff_x0 < 0:
    rotation_angle_rad0 += math.pi # Dodajemo 180 stepeni ako je diff_x negativan

rotation_angle_deg0 = math.degrees(rotation_angle_rad0)
#print("Ugao rotacije: {:.2f} stepeni".format(rotation_angle_deg0))
translation_x0 = 0
translation_y0 = bottom_left_point0[1] - bottom_right_point0[1]
M0 = cv2.getRotationMatrix2D((int(width0/2), int(height0/2)), rotation_angle_deg0, 1.0)
M0[0, 2] += translation_x0
M0[1, 2] += translation_y0
rotated0 = cv2.warpAffine(img0, M0, (width0, height0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#cv2.imshow("Originalna slika Sablona", img0)
#cv2.imshow("Rotirana i translirana slika Sablon", rotated0)

#----------------------------------------------------------------Obradjivanje rotirane slike  Sablona-----------------------------------------------------------------------------
# Učitavanje slike
img =rotated0
original = img.copy()
# Pretvaranje u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Primjena bilateralnog filtra za uklanjanje šuma
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Primjena adaptivne binarizacije
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Pronalazak kontura
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Pronalazak najveće i druge najveće konture
max_contour = None
second_max_contour = None
max_area = 0
second_max_area = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        second_max_area = max_area
        max_area = area
        second_max_contour = max_contour
        max_contour = contour
    elif area > second_max_area:
        second_max_area = area
        second_max_contour = contour
br=0
koordinataS0=[];
# Crtanje okvira druge najveće konture
if second_max_contour is not None:
    x, y, w, h = cv2.boundingRect(second_max_contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #print("S_1Dimenzije druge najveće konture: {} x {}".format(w, h))
    #print("S_1Koordinate druge najveće konture: X{}  Y{}".format(x, y+h))
    koordinataS0.append(x)
    koordinataS0.append(y+h)
#print(koordinataS0)
br1=0
konture=[];
inner_contours = []
nizPiksela1=[]
json_data1 = {}
# Crtanje okvira druge najveće konture
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if h > 20 and w > 20 and w<90 and h<80:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi1 = original[y:y + h, x:x + w]
        roi1 = cv2.resize(roi1, (800, 800))
        height, width = roi1.shape[:2]
        center_x, center_y = int(width/2), int(height/2)
        square_size = int(0.5 * min(height, width))
        roi1 = roi1[center_y - square_size // 2:center_y + square_size // 2, center_x - square_size // 2:center_x + square_size // 2]
        num_black_pixels1 = np.count_nonzero(roi1 == 0)
        br1=br1+1
        nizPiksela1.append(num_black_pixels1)
        #print("S1:Broj crnih piksela na slici konture {}: {}".format(br1, num_black_pixels1))
        roi1 = cv2.resize(roi1, (40, 40))
        #cv2.imwrite(os.path.join('S1_kontura2_{}.png'.format(br1)), roi1)
        #print("Koordinate {} pronadjene konture su: X{} Y{}".format(br1, x,  y))
        cv2.putText(img, "{}.".format(br1), (x +15, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        inner_contours.append(contour)
#================================================================================================================================================================================#

#=================================================FUNKCIJA ODABERI SLIKU ISPRAVNOG TESTA==========================================================================================#    
        
#=================================================Ucitavanje prve slike i rotiranje=========================================================#
niz1=[];
def izaberi_jednu_sliku():
    global niz1
    file_path = filedialog.askopenfilename(filetypes=[("Slike", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Ovde možete obraditi izabranu sliku kako želite
        print("Izabrana slika:", file_path)
        img1S = cv2.imread(file_path)
       # img1S = cv2.resize(img1S, (715, 1010))
        
        # Dobijanje dimenzija slike
        height1S, width1S, _ = img1S.shape

        # Konvertovanje u grayscale
        gray1S = cv2.cvtColor(img1S, cv2.COLOR_BGR2GRAY)

        # Binarizacija
        _1S, thresh1S = cv2.threshold(gray1S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Pronalazak kontura
        contours1S, hierarchy1S = cv2.findContours(thresh1S, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Pronalazak najveće konture
        max_contour1S = max(contours1S, key=cv2.contourArea)

        # Aproksimacija konture
        epsilon1S = 0.1 * cv2.arcLength(max_contour1S, True)
        approx1S = cv2.approxPolyDP(max_contour1S, epsilon1S, True)

        # Definisanje boja tačaka u odgovarajućem redosledu
        colors1S = OrderedDict([('zelena', (0, 255, 0)),
                            ('plava', (255, 0, 0)),
                            ('zuta', (0, 255, 255)),
                            ('braon', (42, 42, 165))])

        # Obeležavanje tačaka sa odgovarajućom bojom
        for i, point in enumerate(approx1S):
            # Provera položaja tačke na slici
            if point[0][1] > height1S / 2:  # Donji deo slike
                if i % 2 == 0:
                    color_name1S = 'plava'
                else:
                    color_name1S = 'zuta'
            else:  # Gornji deo slike
                if i % 2 == 0:
                    color_name1S = 'braon'
                else:
                    color_name1S = 'zelena'

            color1S = colors1S[color_name1S]

            # Ispisivanje vrednosti i značenja tačaka
            #print("Koordinate tačke {}: {}".format(i+1, tuple(point[0])))
            #print("Boja tačke {}: {}".format(i+1, color_name0))

        n1S = len(approx1S)
        slopes1S = []  # Prazna lista za čuvanje nagiba

        bottom_left_index1S = [i for i, point in enumerate(approx1S) if point[0][0] < width1S / 2 and point[0][1] > height1S / 2][0]
        bottom_left_point1S = approx1S[bottom_left_index1S][0]

        #cv2.circle(img0, tuple(bottom_left_point0), 5, (0, 0, 255), -1)
        #print("Koordinate donje leve tačke: ({}, {})".format(bottom_left_point0[0], bottom_left_point0[1]))

        bottom_right_index1S = [i for i, point in enumerate(approx1S) if point[0][0] > width1S / 2 and point[0][1] > height1S / 2][0]
        bottom_right_point1S = approx1S[bottom_right_index1S][0]

        diff_x1S = bottom_right_point1S[0] - bottom_left_point1S[0]
        diff_y1S = bottom_right_point1S[1] - bottom_left_point1S[1]
        rotation_angle_rad1S = math.atan(diff_y1S / diff_x1S)

        if diff_x1S < 0:
            rotation_angle_rad1S += math.pi # Dodajemo 180 stepeni ako je diff_x negativan

        rotation_angle_deg1S = math.degrees(rotation_angle_rad1S)
        #print("Ugao rotacije: {:.2f} stepeni".format(rotation_angle_deg0))
        translation_x1S = 0
        translation_y1S = bottom_left_point1S[1] - bottom_right_point1S[1]
        M1S = cv2.getRotationMatrix2D((int(width1S/2), int(height1S/2)), rotation_angle_deg1S, 1.0)
        M1S[0, 2] += translation_x1S
        M1S[1, 2] += translation_y1S
        rotated1S = cv2.warpAffine(img1S, M1S, (width1S, height1S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #cv2.imshow("Originalna slika Sablona", img0)
        #cv2.imshow("Rotirana i translirana slika Sablon", rotated0)



        #-----------------------------------------------------Obradjivanje prve slike nakon rotiranja--------------------------------------------------------------------------------

        # Učitavanje nove slike
        img1 = rotated1S
        #img1 = cv2.resize(img1, (715, 1010))#Podesavanje dimenzija slike 
        original1=img1.copy()

        # Pretvaranje u grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Primjena bilateralnog filtra za uklanjanje šuma
        gray1 = cv2.bilateralFilter(gray1, 11, 17, 17)



        # Primjena adaptivne binarizacije
        thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Pronalazak kontura
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Pronalazak druge najvece konutre
        max_contour1 = None
        second_max_contour1 = None
        max_area1 = 0
        second_max_area1 = 0

        for contour in contours1:
            area = cv2.contourArea(contour)
            if area > max_area1:
                second_max_area1 = max_area1
                max_area1 = area
                second_max_contour1 = max_contour1
                max_contour1 = contour
            elif area > second_max_area1:
                second_max_area1 = area
                second_max_contour1 = contour
     
        koordinataS1=[];
        # Crtanje okvira druge najveće konture
        if second_max_contour1 is not None:
            x, y, w, h = cv2.boundingRect(second_max_contour1)
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print("S2_Dimenzije druge najveće konture: {} x {}".format(w, h))
            #print("S_2Koordinate druge najveće konture: X{}  Y{}".format(x, y+h))
            koordinataS1.append(x)
            koordinataS1.append(y+h)
        print(koordinataS1)
        #cv2.imshow('Druga najveća kontura druge slike', img2)



        #Racunanje koordinata koje se koriste za transliranje (dobijene druge rotirane slike) u odnosu na prvu
        stelovanje1=[]
        for i in range(2):
            stelovanje1.append(koordinataS0[i]-koordinataS1[i]);
        print(stelovanje1)
        contours1 = sorted(contours1, key=lambda c: (cv2.boundingRect(c)[1]))
        delta_x1 = koordinataS0[0] - koordinataS1[0]
        delta_y1 = koordinataS0[1] - koordinataS1[1]
        br2=0;
        rezultati1= np.empty((20,4))
        nizPiksela1=[];
        red1=0
        kolona1=0

        # Pomeranje kontura sa prve slike na drugu sliku
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20 and w > 20 and w < 90 and h < 80:
                new_x1 = x - delta_x1
                new_y1 = y - delta_y1
                cv2.rectangle(img1, (new_x1+3, new_y1), (new_x1+w-3, new_y1+h-3), (0, 255, 0), 1)
                roi1 = original1[new_y1:new_y1+h-3, new_x1+3: new_x1+w-3]
                roi1 = cv2.resize(roi1, (800, 800))
                height1, width1 = roi1.shape[:2]
                center_x1, center_y1 = int(width1/2), int(height1/2)
                square_size1 = int(0.7* min(height1, width1))
                roi1 = roi1[center_y1 - square_size1 // 2:center_y1 + square_size1 // 2, center_x1 - square_size1 // 2:center_x1 + square_size1 // 2]                
                br2=br2+1
                #print("S2:Broj crnih piksela na slici konture {}: {}".format(br2, num_black_pixels2))
                roi1 = cv2.resize(roi1, (40, 40))
                cv2.imwrite(('roiT78_{}.png'.format(br2)), roi1)
                model = load_model('model.h5')
                # Učitavanje testne slike
                test_image1 = load_img(f'roiT78_{br2}.png', target_size=(40, 40))
                test_image1 = img_to_array(test_image1)
                test_image1 = np.expand_dims(test_image1, axis=0)
                # Predviđanje na testnoj slici
                result1 = model.predict(test_image1)
                prediction1 = np.argmax(result1)                
                if prediction1 == 0:
                    prediction_label1 = 'popunjen'
                elif prediction1 == 1:
                    prediction_label1 = 'prazan'
                else:
                    prediction_label1 = 'neispravno popunjen'
                print(prediction1)
                print(prediction_label1)
                rezultati1[red1, kolona1] = prediction1
                kolona1 += 1  # Pređi na sledeću kolonu
                if kolona1 == 4:  # Ako je popunjena cela kolona, pređi na sledeći red
                    red1 += 1
                    kolona1 = 0

                nizPiksela1.append([{
                    "status kruzica": int(prediction1),
                    "redni broj": br2
                }])
                
                #print("Koordinate {} pronadjene konture su: X{} Y{}".format(br2, x,  y))
                cv2.putText(img1, "{}.".format(br2), (new_x1, new_y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        obrisi_slike() #Brisanje svih slika koje se upisuju u trenutni direktorijum tokom procesa izdvajanja kruzica

        print(rezultati1)
        #print("prvih 10 elemenata matrice")
        print(rezultati1[:10])
        #print("Obrnuta matrica")
        rezultati_unazad1=np.fliplr(rezultati1)
        print(rezultati_unazad1)


            #niz1=[];
            # Kreiranje json fajla

        br2=0;
    

        # Prikazivanje slike s označenim pronađenim konturama
        #cv2.imshow('Druga najveća kontura rotirane slike sablona', img)
        #cv2.imshow('Konture na drugoj slici', img2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    #=============================================Trazenje nepraznih kruzica prve slike=============================================#
    filtered_data1=[]
    json_data1 = {"rezultati": nizPiksela1}
    with open('Svi_odgovori.json', 'w') as f:
        json.dump(json_data1, f, indent=4)
    # Učitavanje podataka iz JSON fajla
    with open('Svi_odgovori.json', 'r') as f:
        json_data1 = json.load(f)
    # Filtriranje elemenata
    filtered_data1 = []
    for sublist in json_data1["rezultati"]:
        for item in sublist:
            if item["status kruzica"] != 1 :
                filtered_data1.append(item)
    # Kreiranje novog JSON objekta
    new_json_data1 = {"rezultati": filtered_data1}
    # Snimanje novog JSON fajla
    with open('Odgovori_razliciti_od_1.json', 'w') as f:
        json.dump(new_json_data1, f, indent=4)
    # Sortiranje po rednom broju slike
    with open('Odgovori_razliciti_od_1.json', 'r') as f:
        json_data1 = json.loads(f.read())
    sorted_data1 = sorted(json_data1["rezultati"], key=lambda x: x["redni broj"])
    sorted_json1 = {"rezultati": sorted_data1}
    with open('Odgovori_razliciti_od_1.json', 'w') as f:
        json.dump(sorted_json1, f, indent=4)
    # Ubacivanje rednih brojeva odgovora u niz
    niz1 = [element['redni broj'] for element in sorted_data1]
    print(niz1)
  
#===================================================================================================================================================================================#

#============================================================================FUNKCIJA ODABERI VISE SLIKA============================================================================#
#============================================================================BIRANJE TESTOVA KOJE SU POPUNILI KANDIDATI=============================================================#
def izaberi_vise_slika():
    global ime_slike
    file_paths = filedialog.askopenfilenames(filetypes=[("Slike", "*.png;*.jpg;*.jpeg")])
    if file_paths:
        # Ovde ćemo smestiti izabrane slike u rečnik
        for index, file_path in enumerate(file_paths):
            image_id = index + 1
            image_name = file_path.split("/")[-1]
            slike_dict[image_id] = {"putanja": file_path, "ime_slike": image_name}
            print("\nPutanje do izabranih slika:")
            
        for image_id, image_info in slike_dict.items():
            ime_slike = image_info["ime_slike"]
            print(ime_slike)
            print(image_info['putanja'])
            img2S = cv2.imread(image_info['putanja'])
           # img2S = cv2.resize(img2S, (715, 1010))
            
            # Dobijanje dimenzija slike
            height2S, width2S, _ = img2S.shape

            # Konvertovanje u grayscale
            gray2S = cv2.cvtColor(img2S, cv2.COLOR_BGR2GRAY)

            # Binarizacija
            _2S, thresh2S = cv2.threshold(gray2S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Pronalazak kontura
            contours2S, hierarchy2S = cv2.findContours(thresh2S, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Pronalazak najveće konture
            max_contour2S = max(contours2S, key=cv2.contourArea)

            # Aproksimacija konture
            epsilon2S = 0.1 * cv2.arcLength(max_contour2S, True)
            approx2S = cv2.approxPolyDP(max_contour2S, epsilon2S, True)

            # Definisanje boja tačaka u odgovarajućem redosledu
            colors2S = OrderedDict([('zelena', (0, 255, 0)),
                                ('plava', (255, 0, 0)),
                                ('zuta', (0, 255, 255)),
                                ('braon', (42, 42, 165))])

            # Obeležavanje tačaka sa odgovarajućom bojom
            for i, point in enumerate(approx2S):
                # Provera položaja tačke na slici
                if point[0][1] > height2S / 2:  # Donji deo slike
                    if i % 2 == 0:
                        color_name2S = 'plava'
                    else:
                        color_name2S = 'zuta'
                else:  # Gornji deo slike
                    if i % 2 == 0:
                        color_name2S = 'braon'
                    else:
                        color_name2S = 'zelena'

                color2S = colors2S[color_name2S]

                # Ispisivanje vrednosti i značenja tačaka
                #print("Koordinate tačke {}: {}".format(i+1, tuple(point[0])))
                #print("Boja tačke {}: {}".format(i+1, color_name0))

            n2S = len(approx2S)
            slopes2S = []  # Prazna lista za čuvanje nagiba

            bottom_left_index2S = [i for i, point in enumerate(approx2S) if point[0][0] < width2S / 2 and point[0][1] > height2S / 2][0]
            bottom_left_point2S = approx2S[bottom_left_index2S][0]

            #cv2.circle(img0, tuple(bottom_left_point0), 5, (0, 0, 255), -1)
            #print("Koordinate donje leve tačke: ({}, {})".format(bottom_left_point0[0], bottom_left_point0[1]))

            bottom_right_index2S = [i for i, point in enumerate(approx2S) if point[0][0] > width2S / 2 and point[0][1] > height2S / 2][0]
            bottom_right_point2S = approx2S[bottom_right_index2S][0]

            diff_x2S = bottom_right_point2S[0] - bottom_left_point2S[0]
            diff_y2S = bottom_right_point2S[1] - bottom_left_point2S[1]
            rotation_angle_rad2S = math.atan(diff_y2S / diff_x2S)

            if diff_x2S < 0:
                rotation_angle_rad2S += math.pi # Dodajemo 180 stepeni ako je diff_x negativan

            rotation_angle_deg2S = math.degrees(rotation_angle_rad2S)
            #print("Ugao rotacije: {:.2f} stepeni".format(rotation_angle_deg0))
            translation_x2S = 0
            translation_y2S = bottom_left_point2S[1] - bottom_right_point2S[1]
            M2S = cv2.getRotationMatrix2D((int(width2S/2), int(height2S/2)), rotation_angle_deg2S, 1.0)
            M2S[0, 2] += translation_x2S
            M2S[1, 2] += translation_y2S
            rotated2S = cv2.warpAffine(img2S, M2S, (width2S, height2S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            #cv2.imshow("Originalna slika Sablona", img0)
            #cv2.imshow("Rotirana i translirana slika Sablon", rotated0)



            #-----------------------------------------------------Obradjivanje druge slike nakon rotiranja--------------------------------------------------------------------------------

            # Učitavanje nove slike
            img2 = rotated2S
           # img2 = cv2.resize(img2, (715, 1010))#Podesavanje dimenzija slike 
            original2=img2.copy()

            # Pretvaranje u grayscale
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Primjena bilateralnog filtra za uklanjanje šuma
            gray2 = cv2.bilateralFilter(gray2, 11, 17, 17)

            # Primjena adaptivne binarizacije
            thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Pronalazak kontura
            contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Pronalazak druge najvece konutre
            max_contour2 = None
            second_max_contour2 = None
            max_area2 = 0
            second_max_area2 = 0

            for contour in contours2:
                area = cv2.contourArea(contour)
                if area > max_area2:
                    second_max_area2 = max_area2
                    max_area2 = area
                    second_max_contour2 = max_contour2
                    max_contour2 = contour
                elif area > second_max_area2:
                    second_max_area2 = area
                    second_max_contour2 = contour
            br=0
            br1=0
            koordinataS2=[];
            # Crtanje okvira druge najveće konture
            if second_max_contour2 is not None:
                x, y, w, h = cv2.boundingRect(second_max_contour2)
                cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #print("S2_Dimenzije druge najveće konture: {} x {}".format(w, h))
                #print("S_2Koordinate druge najveće konture: X{}  Y{}".format(x, y+h))
                koordinataS2.append(x)
                koordinataS2.append(y+h)
            print(koordinataS2)
            #cv2.imshow('Druga najveća kontura druge slike', img2)

            #Racunanje koordinata koje se koriste za transliranje (dobijene druge rotirane slike) u odnosu na prvu
            stelovanje2=[]
            for i in range(2):
                stelovanje2.append(koordinataS0[i]-koordinataS2[i]);
            print(stelovanje2)
            contours2 = sorted(contours2, key=lambda c: (cv2.boundingRect(c)[1]))          
            delta_x2 = koordinataS0[0] - koordinataS2[0]
            delta_y2 = koordinataS0[1] - koordinataS2[1]
            br2=0;
            rezultati2= np.empty((20,4))
            nizPiksela2=[];
            red2=0
            kolona2=0

            # Pomeranje kontura sa prve slike na drugu sliku
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if h > 20 and w > 20 and w < 90 and h < 80:
                    new_x2 = x - delta_x2
                    new_y2 = y - delta_y2
                    cv2.rectangle(img2, (new_x2+3, new_y2), (new_x2+w-3, new_y2+h-3), (0, 255, 0), 1)
                    roi2 = original2[new_y2:new_y2+h-3, new_x2+3: new_x2+w-3]
                    roi2 = cv2.resize(roi2, (800, 800))
                    height2, width2 = roi2.shape[:2]
                    center_x2, center_y2 = int(width2/2), int(height2/2)
                    square_size2 = int(0.7* min(height2, width2))
                    roi2 = roi2[center_y2 - square_size2 // 2:center_y2 + square_size2 // 2, center_x2 - square_size2 // 2:center_x2 + square_size2 // 2]                    
                    br2=br2+1
                    #print("S2:Broj crnih piksela na slici konture {}: {}".format(br2, num_black_pixels2))
                    roi2 = cv2.resize(roi2, (40, 40))
                    cv2.imwrite(os.path.join('roiT78_{}.png'.format(br2)), roi2)
                    model = load_model('model.h5')
                    # Učitavanje testne slike
                    test_image2 = load_img(f'roiT78_{br2}.png', target_size=(40, 40))
                    test_image2 = img_to_array(test_image2)
                    test_image2 = np.expand_dims(test_image2, axis=0)
                    # Predviđanje na testnoj slici
                    result2 = model.predict(test_image2)
                    prediction2 = np.argmax(result2)                 
                    if prediction2 == 0:
                        prediction_label2 = 'popunjen'
                    elif prediction2 == 1:
                        prediction_label2 = 'prazan'
                    else:
                        prediction_label2 = 'neispravno popunjen'
                    print(prediction2)
                    print(prediction_label2)
                    rezultati2[red2, kolona2] = prediction2
                    kolona2 += 1  # Pređi na sledeću kolonu
                    if kolona2 == 4:  # Ako je popunjena cela kolona, pređi na sledeći red
                        red2 += 1
                        kolona2 = 0

                    nizPiksela2.append([{
                        "status kruzica": int(prediction2),
                        "redni broj": br2
                    }])
                    
                    #print("Koordinate {} pronadjene konture su: X{} Y{}".format(br2, x,  y))
                    cv2.putText(img2, "{}.".format(br2), (new_x2, new_y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            obrisi_slike() #Brisanje svih slika koje se upisuju u trenutni direktorijum tokom procesa izdvajanja kruzica
            print(rezultati2)
            #print("prvih 10 elemenata matrice")
            print(rezultati2[:10])
            #print("Obrnuta matrica")
            rezultati_unazad=np.fliplr(rezultati2)
            print(rezultati_unazad)


            niz2=[];
                # Kreiranje json fajla
            json_data2 = {}
                # Pronalaženje odgovarajućih kontura
            br2=0;
            

            # Prikazivanje slike s označenim pronađenim konturama
            #cv2.imshow('Druga najveća kontura rotirane slike sablona', img)
            #cv2.imshow('Konture na drugoj slici', img2)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


            #=============================================Trazenje nepraznih kruzica druge slike=============================================#
            filtered_data2=[]
            json_data2 = {"rezultati": nizPiksela2}
            with open('Svi_odgovori.json', 'w') as f:
                json.dump(json_data2, f, indent=4)
            # Učitavanje podataka iz JSON fajla
            with open('Svi_odgovori.json', 'r') as f:
                json_data2 = json.load(f)
            # Filtriranje elemenata
            filtered_data2 = []
            for sublist in json_data2["rezultati"]:
                for item in sublist:
                    if item["status kruzica"] != 1 :
                        filtered_data2.append(item)
            # Kreiranje novog JSON objekta
            new_json_data2 = {"rezultati": filtered_data2}
            # Snimanje novog JSON fajla
            with open('Odgovori_razliciti_od_1.json', 'w') as f:
                json.dump(new_json_data2, f, indent=4)
            # Sortiranje po rednom broju slike
            with open('Odgovori_razliciti_od_1.json', 'r') as f:
                json_data2 = json.loads(f.read())
            sorted_data2 = sorted(json_data2["rezultati"], key=lambda x: x["redni broj"])
            sorted_json2 = {"rezultati": sorted_data2}
            with open('Odgovori_razliciti_od_1.json', 'w') as f:
                json.dump(sorted_json2, f, indent=4)
            # Ubacivanje rednih brojeva odgovora u niz
            niz2 = [element['redni broj'] for element in sorted_data2]
            print(niz2)
            
            
            #=================================================================================================================================================#
            #listaN neutrakni odgovori
            listaN=[77,73,69,65,61,57,53,49,45,41,37,33,29,25,21,17,13,9,5,1];
            #odgovori predstavlja cetiri ponudjena odgovora svakog od 20 pitanja
            odgovori=[
                    [77,78,79,80],[73,74,75,76],[69,70,71,72],[65,66,67,68],
                    [61,62,63,64],[57,58,59,60],[53,54,55,56],[49,50,51,52],
                    [45,46,47,48],[41,42,43,44],[37,38,39,40],[33,34,35,36],
                    [29,30,31,32],[25,26,27,28],[21,22,23,24],[17,18,19,20],
                    [13,14,15,16],[9,10,11,12],[5,6,7,8],[1,2,3,4]
                    ]
          
            #=====================================Filtriranje odgovora koji nisu validni od valdinih=========================================================================#
            for sublist in odgovori:
                common_elements = list(set(niz2).intersection(set(sublist)))
                if len(common_elements) > 1:
                    print("Pronađeno više od 1 elementa iz Niza1 u podlisti odgovori:", sublist)
                    print("Elementi koji su pronađeni:", common_elements)
                    niz2 = [x for x in niz2 if x not in sublist]   
            print("Niz2 nakon uklanjanja elemenata:", niz2)
            print("-----------------------------------------------------------------------------------------------")        
            #==================================================================================================================================================================#

            #===================================Filtriranje kruzica tako da se dobijuu samo oni koji su validni================================================================#

            # Učitavanje JSON fajla
            with open('Odgovori_razliciti_od_1.json', 'r') as f:
                json_data2 = json.load(f)

            # Filtriranje JSON fajla
            filtered_rezultati2 = [element for element in json_data2["rezultati"] if element["redni broj"] in niz2]

            # Ažuriranje JSON objekta sa filtriranim podacima
            json_data2["rezultati"] = filtered_rezultati2

            # Ažuriranje JSON fajla sa izmenjenim podacima
            with open('Filtrirani_odgovori_razliciti_od_1.json', 'w') as f:
                json.dump(json_data2, f, indent=4)

            with open('Filtrirani_odgovori_razliciti_od_1.json', 'r') as f:
                json_data2 = json.load(f)
            # Filtriranje elemenata
            filtered_data2 = []
            for item in json_data2["rezultati"]:
                if item["status kruzica"] == 0:
                    filtered_data2.append(item)

            # Kreiranje novog JSON objekta
            new_json_data2 = {"rezultati": filtered_data2}
            # Snimanje novog JSON fajla
            with open('Vazeci_odgovori.json', 'w') as f:
                json.dump(new_json_data2, f, indent=4)

            with open('Vazeci_odgovori.json', 'r') as f:
                json_data2 = json.loads(f.read())
            sorted_data2 = sorted(json_data2["rezultati"], key=lambda x: x["redni broj"])
            sorted_json2 = {"rezultati": sorted_data2}
            with open('Vazeci_odgovori.json', 'w') as f:
                json.dump(sorted_json2, f, indent=4)
                # Ubacivanje rednih brojeva odgovora u niz
            niz2 = [element['redni broj'] for element in sorted_data2]
            print("Niz2 nakon uklanjanja elemenata spreman za obradu poena:", niz2)
            #===========================================================================================================================================================#

           
            tac=0;
            neu=0;
            neT=0
            if len(niz1) > len(niz2):
                print(f"Kandidat se ne prizanju svih 20 pitanja");

                print("-----------------------------------------------------------------------------------------------")
                
                nizPogresni=[]
                for index, sublist in enumerate(odgovori, start=1):
                    neutral_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                    for element in sublist:
                        if element in niz2 and element in listaN:
                            neutral_odgovoreno = True
                            niz2.remove(element)
                            break
                    if neutral_odgovoreno:
                        print(f"Kandidat je odgovorio sa ne znam na {index} pitanje")
                        nizPogresni.append(index)
                        neu=neu+1
                
                print("-----------------------------------------------------------------------------------------------")
                print("Niz nakon sto se izbace pitanja na koja je kandidat odgovorio sa ne znam")
                print(niz2)
                print("-----------------------------------------------------------------------------------------------")
                for index, sublist in enumerate(odgovori,start=1):
                    if index>20:
                        break; 
                    tacno_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                    for element in sublist:
                        if element in niz2 and element in niz1:
                            tacno_odgovoreno = True
                            niz2.remove(element)
                            break
                    if tacno_odgovoreno:
                        print("Kandidat je tačno odgovorio na pitanje", index)
                        nizPogresni.append(index)
                        tac=tac+1
                print("-----------------------------------------------------------------------------------------------")  
                print("Niz nakon sto se izbace pitanja na koja je kandidat odgovorio tacno")
                print(niz2)    
                for index, sublist in enumerate(odgovori,start=1):
                    if index>20:
                        break; 
                    netacno_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                    for element in sublist:
                        if element in niz2:
                            netacno_odgovoreno = True
                            break
                    if netacno_odgovoreno:
                        print("Kandidat je netačno odgovorio na pitanje", index)
                        nizPogresni.append(index)
                        neT=neT+1
                print("-----------------------------------------------------------------------------------------------")
                print("Pitanja na koje se prizanje odgovor:",nizPogresni)
                for i in reversed(range(1, 21)):
                    if i not in nizPogresni:
                        print(f"Kandidatu se ne priznaje {i} pitanje.")
                
                print(f"Kandidau se ne prizanje odgovor na:{20-len(nizPogresni)} pitanja")
                print(f"Tacni odgovori:{tac}")
                print(f"Neutralni odgovori:{neu}")
                print(f"Netacni odgovori:{neT}")
                bodovi=tac*5+neu*0+neT*(-1);

                #Ubacivanje u tabelu
                popuni_tabelu(ime_slike,bodovi)
                if bodovi>0:
                    print(f"Kandidat je ostvario: {bodovi} poena na prijemnom.");
                else:
                    print(f"Kandidat je ostvario: 0 poena na prijemnom.");



            elif len(niz1)==len(niz2):
                print(f"Kandidat se prizanju svih 20 pitanja");
                print("-----------------------------------------------------------------------------------------------")
                #nizPogresni2=[]
                tacan=0;
                neutral=0;
                for index, sublist in enumerate(odgovori, start=1):
                    neutral_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                    for element in sublist:
                        if element in niz2 and element in listaN:
                            neutral_odgovoreno = True
                            niz2.remove(element)
                            break
                    if neutral_odgovoreno:
                        print(f"Kandidat je odgovorio sa ne znam na {index} pitanje")
                        #nizPogresni2.append(index)
                        neutral=neutral+1

                print("-----------------------------------------------------------------------------------------------")
                
                for index, sublist in enumerate(odgovori,start=1):
                        if index>20:
                            break; 
                        tacno_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                        for element in sublist:
                            if element in niz2 and element in niz1:
                                tacno_odgovoreno = True
                                niz2.remove(element)
                                break
                        if tacno_odgovoreno:
                            print("Kandidat je tačno odgovorio na pitanje", index)
                            #nizPogresni2.append(index)
                            tacan=tacan+1
                print("-----------------------------------------------------------------------------------------------")
                netacan=0;
                for index, sublist in enumerate(odgovori,start=1):
                        if index>20:
                            break; 
                        netacno_odgovoreno = False  # Promenljiva za praćenje da li je kandidat tačno odgovorio na pitanje
                        for element in sublist:
                            if element in niz2:
                                netacno_odgovoreno = True
                                break
                        if netacno_odgovoreno:
                            print("Kandidat je netačno odgovorio na pitanje", index)
                            #nizPogresni2.append(index)
                            netacan=netacan+1
                print("-----------------------------------------------------------------------------------------------")
                print(f"Tacni odogovri:{tacan}")
                print(f"Neutralni odogovri:{neutral}")
                print(f"Netacni odogovri:{netacan}")
                global poeni;
                poeni=tacan*5+netacan*(-1)+neutral*0;

                #Ubacivanje u tabelu
                popuni_tabelu(ime_slike,poeni)
                if poeni>0:
                                print(f"Kandidat je ostvario :{poeni} poena na prijemnom")
                else:
                                print(f"Kandidat je ostvario : 0 poena na prijemnom") 
    obrisi_json_fajlove()
#============================================================================================================================================================================#    


#============================================FUNKCIJA KOJA POPUNJAVA TABELU I BAZU PODATAKA ==================================================================================#
def popuni_tabelu(ime_slike, poeni):
    parts = os.path.splitext(ime_slike)[0].split('_')
    ime_sa_razmacima = ' '.join(parts)

    # Povezivanje sa MySQL bazom podataka
    connection = mysql.connector.connect(
        host=os.environ.get('MYSQL_HOST', 'localhost'),
        user=os.environ.get('MYSQL_USER', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', 'petarkostic123'),
        database=os.environ.get('MYSQL_DATABASE', 'testovi_db')
    )
    cursor = connection.cursor()

    # Definisanje SQL upita za ubacivanje podataka u tabelu baze
    query = "INSERT INTO tabela (Ime_Prezime, Poeni) VALUES (%s, %s)"
    values = (ime_sa_razmacima, poeni)

    # Izvršavanje SQL upita
    cursor.execute(query, values)

    # Potvrda promena i zatvaranje konekcije
    connection.commit()
    connection.close()

    # tabela.insert("", "end", values=(ime_sa_razmacima, poeni))
#=========================================================================================================================================================================#
#==========================================FUNKCIJA KOJA BRISE PODATKE IZ BAZE============================================================================================#

def obrisi_sve_podatke():
    try:
        # Povezivanje sa bazom podataka
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', 'petarkostic123'),
            database=os.environ.get('MYSQL_DATABASE', 'testovi_db')
        )
        
        # Kreiranje SQL upita za brisanje svih podataka iz tabele
        query = "DELETE FROM tabela"
        
        # Izvršavanje SQL upita
        cursor = connection.cursor()
        cursor.execute(query)
        
        # Potvrda promena i zatvaranje konekcije
        connection.commit()
        connection.close()
        
        print("Svi podaci su uspešno obrisani iz tabele.")
    except mysql.connector.Error as error:
        print(f"Došlo je do greške: {error}")
#=================================================GUI=====================================================================================================================#
#==========================================FUNKCIJA KOJA BRISE PODATKE IZ BAZE============================================================================================#
def izlistaj_sve_iz_baze():
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', 'petarkostic123'),
            database=os.environ.get('MYSQL_DATABASE', 'testovi_db')
        )
        cursor = connection.cursor()

        # Dobijanje svih podataka iz tabele
        query = "SELECT Ime_Prezime, Poeni FROM tabela"
        cursor.execute(query)
        result = cursor.fetchall()

        # Ako tabela ima podatke, brišemo postojeće stavke iz liste
        if len(result) > 0:
            tabela.delete(*tabela.get_children())

        # Prolazak kroz dobijene rezultate i popunjavanje tabele
        for row in result:
            ime_prezime, poeni = row
            tabela.insert("", "end", values=(ime_prezime, poeni))

        # Zatvaranje konekcije
        connection.close()
    except mysql.connector.Error as error:
        print(f"Došlo je do greške: {error}")


#==========================================================================================================================================================================#

# Kreiranje glavnog prozora
root = tk.Tk()
root.title("YAP")

# Rečnik u kojem ćemo čuvati informacije o slikama
slike_dict = {}

# Dugme za izbor jedne slike

dugme_izaberi_jednu_sliku = tk.Button(root, text="Izaberi sliku ispravno popunjenog testa", command=izaberi_jednu_sliku)
dugme_izaberi_jednu_sliku.pack(side=tk.LEFT, padx=10, pady=10)

# Dugme za izbor više slika
dugme_izaberi_vise_slika = tk.Button(root, text="Izaberi slike testova kandidata za obradu", command=izaberi_vise_slika)
dugme_izaberi_vise_slika.pack(side=tk.LEFT, padx=10, pady=10)

# Dugme za izlistavanje svega iz baze
dugme_izlistaj_sve_iz_baze = tk.Button(root, text="Izlistaj podatke o kandidatima", command=izlistaj_sve_iz_baze)
dugme_izlistaj_sve_iz_baze.pack(side=tk.LEFT, padx=10, pady=10)

# Dugme za promenu dimenzije slika
dugme_promena_dimenzije_slika = tk.Button(root, text="Promeni dimenzije slika", command=resize_images)
dugme_promena_dimenzije_slika.pack(side=tk.LEFT, padx=10, pady=10)

# Dugme za brisanje tabele iz baze podataka
dugme_iobrisi_iz_baze = tk.Button(root, text="Obrisi podatke iz baze", command=obrisi_sve_podatke)
dugme_iobrisi_iz_baze.pack(side=tk.BOTTOM, padx=10, pady=10)

# Kreiranje tabele sa slikama
tabela = ttk.Treeview(root, columns=("Ime i prezime", "Br poena"))
tabela.heading("#1", text="Ime i prezime")
tabela.heading("#2", text="Br poena")
tabela.pack(padx=10, pady=5)



# Pokretanje glavnog prozora
root.mainloop()


