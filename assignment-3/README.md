# Assignment 3

**Meno:** Ákos Kappel

**Predmet:** Počítačová videnie

**Cvičenie:** Pondelok 11:00

**Cvičiaci:** Ing. Marek Jakab, PhD.

**Použité knižnice:** OpenCV, Matplotlib, Numpy



## 1 Motion tracking

Prvú časť tohto zadania som začal s metódami na sledovanie pohybu objektov.
Vyskúšal som dve metódy, ktoré sú implementované v OpenCV: *sparse* a *dense optical flow*.

### 1.1 Sparse optical flow

Pri tejto metóde som začal s načítaním snímok z videa.
Po načítaní snímky sa po každých pár snímok zavolá funkcia `cv2.goodFeaturesToTrack()`.
Táto funkcia vráti zoznam bodov, ktoré sa považujú za vhodné na sledovanie pohybu objektov.
Interval vyberania bodov je nastaviteľný pomocou parametra `detecting_interval`.
Aby som predišiel nadmernému počtu bodov, ktoré sa majú sledovať, tak som vytvoril premennú `max_tracked_points`, ktorá určuje maximálny počet bodov, ktoré sa dajú naraz sledovať v jednej snímke.
Tieto body sa potom sledujú pomocou funkcie `cv2.calcOpticalFlowPyrLK()`.
Pre každý sledovaný bod si uchovávam jeho zoznam predchádzajúcich pozícií v premennej `trajectories`, aby som vedel vykresliť ich trajektoriu.
Aby som zabránil tomu, že niektoré body skočili na veľmi vzdialené miesta medzi snímkami, tak som vytvoril premennú `max_displacement_error`, ktorá určuje maximálnu vzdialenosť, ktorú môže bod prejsť medzi dvoma snímkami.
Kód pre celý tento postup sa nachádza v súbore [motion_tracking.ipynb](motion_tracking.ipynb) vo funkcií `sparse_optical_flow_v2()`.
Výsledky z tejto metódy sú nasledovné:

https://user-images.githubusercontent.com/56064033/229316910-2a8e27e6-8a6b-4a30-be19-c0cc7d7c1eb2.mp4

https://user-images.githubusercontent.com/56064033/229316968-6f040cda-bc37-417b-a733-36e9936250d0.mp4

https://user-images.githubusercontent.com/56064033/229316841-afa858f0-94f5-41c2-b6f2-fd4075c6e61d.mp4

Na videách môžme vidieť, že sa celkom dobre podarilo sledovať pohyby objektov, ale mnoho sledovaých bodov sa vyskytuje na objektoch v pozadí, ktoré sa nehýbu.
Práve preto som skúsil pridať pri výbere sledovaných bodov aj masku, ktorá určuje, ktoré časti snímky sa majú sledovať.
Táto maska je vytvorená pomocou metódy na určenie pozadia (*Mixture of Gaussians*), ktorú používam aj v ďalšej časti tohto zadania.
Kód pre tento modifikovaný spôsob sledovania pohybu sa nachádza vo funkcií `sparse_optical_flow_v3()`.
Na nasledujúcich troch videách je ukážka výsledkov s pridanou maskou na výber sledovaných bodov:

https://user-images.githubusercontent.com/56064033/229512109-f0ce759e-d0ab-4cdd-b1f5-529365f5f65e.mp4

https://user-images.githubusercontent.com/56064033/229512255-d9f53257-f8a3-45e9-9c74-9b2390179566.mp4

https://user-images.githubusercontent.com/56064033/229512298-0ad1026f-af0b-4ba2-8c60-811105faa6fb.mp4

Na videách je jasne vidieť, že sa po pridaní masky na výber sledovaných bodov výrazne zlepšila kvalita sledovania pohybu objektov.

### 1.2 Dense optical flow

Druhú metódu na sledovanie pohybu som opäť začal s načítaním jednotlivých snímok z videa.
Na každú snímku som aplikoval funkciu `cv2.calcOpticalFlowFarneback()`, ktorá vráti maticu s vektormi pohybu.
Z tejto matice som určil smer a rýchlosť pohybu pre každý bod snímky.
Aby som odstránil nežiadaný šum z pohybu, tak som odfiltroval všetky vektory, ktoré sú menšie ako hodnota v premennej `min_magnitude`.
Takto som získal masku, ktorá určuje, ktoré časti snímky sa pohli.
Pre ďalšie zlepšenie kvality sledovania pohybu som na masku aplikoval morfologické operácie *opening* a *closing*.
Výsledné vektory pohybu sa nachádzajú na pravej časti každej snímky.

Pre detekovanie ohraničujúcich boxov som najprv zistil kontúry na maske.
Potom som pre každú kontúru zistil súradnice najmenšieho obdĺžnika, ktorý ju obklopuje.
Získané boxy som ďalej prefiltroval podľa ich minimálneho obsahu.
Následne som aplikoval funkciu `non_max_suppression()`, ktorá zistí, ktoré boxy sa prekryvajú a z nich vyberie ten s najväčším obsahom.
Takto som získal výsledné boxy, ktoré sa vykreslia okolo pohybujúcich sa objektov.

Celý kód tejto metódy sa nachádza v súbore [motion_tracking.ipynb](motion_tracking.ipynb) vo funkcií `dense_optical_flow()`.
Získané výsledky sú nasledovné:

https://user-images.githubusercontent.com/56064033/229513257-589d4aee-d02c-4cde-b22f-f73458071323.mp4

https://user-images.githubusercontent.com/56064033/229513295-ed031ab9-a0b2-4235-825d-e54d4b6c37f1.mp4

https://user-images.githubusercontent.com/56064033/229513331-62e7f449-b320-427a-ab3b-f4b175c8934f.mp4



## 2 Segmentation

### 2.1 Background subtraction

#### 2.1.1 Accumulated weighted image

https://user-images.githubusercontent.com/56064033/229830230-3cd34ee7-2f80-4296-b766-bdbfab45ca54.mp4

https://user-images.githubusercontent.com/56064033/229830439-7184f68a-d723-4cae-b6f8-07b660091b6a.mp4

https://user-images.githubusercontent.com/56064033/229830574-a4f9eb17-a027-41a8-bffe-51929f9369eb.mp4


#### 2.1.2 Mixture of Gaussian

https://user-images.githubusercontent.com/56064033/229830282-c00b46e1-3fa7-4f9b-956c-202870917349.mp4

https://user-images.githubusercontent.com/56064033/229830533-d56fd738-b54f-4202-91cc-5bc31cc11f0c.mp4

https://user-images.githubusercontent.com/56064033/229830610-7b3d5b42-8f1c-4b38-97e0-d7269384d456.mp4


### 2.2 GrabCut

![grabcut 1](output/grabcut-1.png)
![grabcut 1](output/grabcut-2.png)
![grabcut 1](output/grabcut-3.png)
![grabcut 1](output/grabcut-4.png)
![grabcut 1](output/grabcut-5.png)


### 2.3 VOC12

