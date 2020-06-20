import cv2
import time
import f_utils as fu

# crear el detector de rostros frontal
detect_frontal_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
# crear el detector de perfil rostros
detect_perfil_face = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

# visualizar
cv2.namedWindow("preview")
cam = cv2.VideoCapture(0)
while True:
    # read the frame from the camera and send it to the server
    star_time = time.time()
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    #-------------------------- Insertar preproceso -------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectar si hay un rostro frontal o de perfil
    boxes =  fu.detect(gray,detect_frontal_face)
    if len(boxes)!=0:
        tag = 'frontal'
    else:
        boxes,tag = fu.models_profile(gray,detect_perfil_face)
    frame = fu.print_image(frame,boxes,tag)
    # ----------------------------------------------------------------------------
    end_time = time.time() - star_time    
    FPS = 1/end_time
    cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow('preview',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break