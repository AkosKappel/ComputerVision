# Assignment 2

**Meno:** Ákos Kappel

**Predmet:** Počítačová videnie

**Cvičenie:** Pondelok 11:00

**Cvičiaci:** Ing. Marek Jakab, PhD.

**Použité knižnice:** OpenCV, Matplotlib, Numpy


## 1. Experimenty s korekciou obrazu (image correction)

### Farebné systémy

Riešenie druhého zadania som začal s vizualizáciou farebných kanálov pre rôzne farebné systémy.
Na začiatok som načítal obraz s dátovým typom *uint8*, ktorý som následne prekonvertoval na *float32* a hodnoty pixelov som normalizoval do rozsahu [0-1].

```python
# Load the image
img = cv.imread(file)

# Convert to 32-bit float format and normalize to [0, 1]
float32_img = img.astype(np.float32) / 255.0
```

Potom som pomocou funkcie `cv.cvtColor()` prekonvertoval obraz do rôznych farebných kanálov.
Pre každý kanál obrazu som vytvoril aj histogram na vizualizáciu rozloženia hodnôt v danom kanáli.
Na rozdelenie obrazu na jednotlivé kanály som použil funkciu `cv.split()` a na výpočet histogramu som použil funkciu `cv.calcHist()`.

```python
# Convert the image to selected color space
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Split the image into 3 channels
H, S, V = cv.split(hsv_img)

# Calculate histogram for each channel
hist_hue = cv.calcHist([H], [0], None, [180], [0, 180])
hist_sat = cv.calcHist([S], [0], None, [256], [0, 256])
hist_val = cv.calcHist([V], [0], None, [256], [0, 256])
```

Tento postup som zopakoval pre farebné systémy **RGB**, **CIE XYZ**, **Lab**, **HSV**, **HLS** a **YCrCb**.
Jednotlivé kanály a histogramy som potom zobrazil pomocou knižnice **Matplotlib**.
Tu môžeme vidieť niektoré výsledky:

#### Grayscale
![Grayscale image and histogram](out/grayscale.png)
#### RGB
![RGB image and histogram](out/RGB.png)
#### XYZ
![XYZ image and histogram](out/XYZ.png)
#### Lab
![Lab image](out/Lab.png)
![Lab histogram](out/Lab_hist.png)
#### YCrCb
![YCrCb image](out/YCrCb.png)
![YCrCb histogram](out/YCrCb_hist.png)

Všetky ostatné výsledky farebných systémov sa nachádzajú v súbore [image_correction.ipynb](image_correction.ipynb).


### Equalizácia histogramu

V rámci experimentovania s obrazom som vyskúšal aj metódu na equalizáciu histogramu.
Obraz som pred equalizáciou previedol do šedotónového obrazu a potom som použil funkciu `cv.equalizeHist()` na vyrovnanie histogramu.
Na dolných obrázkoch môžeme vidieť obraz a jeho histogram pred a po equalizácii.

![Equalized](out/hist_equalized.png)
![Histogram equalization](out/hist_equalization.png)


### Gamma korekcia

Vyskúšal som aj gamma korekciu obrazu, ktorú som implementoval pomocou funkcie `cv.pow()`.

```python
# Set gamma value
gamma = 1.5

# Apply gamma correction
corrected_img = cv.pow(img / float(np.max(img)), gamma)
```

Na obrázku nižšie je niekoľko ukážok gamma korekcie s rôznymi hodnotami gammy.

![Gamma correction](out/gamma.png)


## 2. Experimenty s úpravou obrazu (image enhancement)


### Segmentácia s využitím informácií o farbách


### Lokalizácia časti obrazu s použitím detektorov a deskriptorov


