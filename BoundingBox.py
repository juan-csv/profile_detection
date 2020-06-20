'''
esta funcion pinta una caja sobre las coordenadas dadas 
Input:
    - img: imagen (array)
    - box: [[y0,x1,y1,x0],[y0,x1,y1,x0],...,[y0,x1,y1,x0]], es una lista de listas con todas las cajas que se quieren pintar
    - match_name: [[name],[name],...,[name]], lista de listas con cada uno de los nombres que se le quiere colocar a la caja, 
        no es un parametro obligatorio
'''

import cv2
import numpy as np

def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

#---------------------------------- ejemplo de consimo ---------------------------------
'''
box = [[353, 123, 710, 480]]
im = cv2.imread("juan.jpg")
im_box = bounding_box(im,box)

cv2.imshow('ejemplo',im_box)
cv2.waitKey(0)
'''
