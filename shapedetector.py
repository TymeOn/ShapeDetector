import cv2
import numpy as np
from matplotlib import pyplot as plt

# lecture de l'image
img = cv2.imread('shapes.png')

# conversion de l'image en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# application d'un seuillage d'image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# utilisation de la fonction findContours()
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

first_match = True

# itération sur la liste des formes trouvées
for contour in contours:

    # la premiere forme est ignorée (c'est l'image elle-même)
    if first_match:
        first_match = False
        continue

    # approximation de la forme 
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # affichage du contour détecté sur l'image
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # calcul du centre de la forme
    M = cv2.moments(contour) 
    if M['m00'] != 0.0: 
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # affichage du nom de la forme détectée, basé sur le nombre de sommets
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 4:
        cv2.putText(img, 'Quadrilatère', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 5:
        cv2.putText(img, 'Pentagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 6:
        cv2.putText(img, 'Hexagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 7:
        cv2.putText(img, 'Heptagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 8:
        cv2.putText(img, 'Octogone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 9:
        cv2.putText(img, 'Ennéagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 10:
        cv2.putText(img, 'Décagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else: 
        cv2.putText(img, 'Cercle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# affichage du résultat final
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
