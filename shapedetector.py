import cv2
import sys

def shapeDetector(filename):
    # lecture de l'image
    img = cv2.imread(filename)

    # conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # application d'un seuillage d'image
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    blurred = cv2.GaussianBlur(threshold, (3, 3), 0)

    # utilisation de la fonction findContours()
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # itération sur la liste des formes trouvées
    for contour in contours:

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
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = 'Carre' if ar >= 0.95 and ar <= 1.05 else "Rectangle"
            cv2.putText(img, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'Pentagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'Hexagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 7:
            cv2.putText(img, 'Heptagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 8:
            cv2.putText(img, 'Octogone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 9:
            cv2.putText(img, 'Enneagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif len(approx) == 10:
            cv2.putText(img, 'Decagone', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.putText(img, 'Cercle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # affichage du résultat final
    cv2.imshow('shapes', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) > 1:
        shapeDetector(sys.argv[1])
    else:
        shapeDetector('input/shapes.png')
