import cv2
import BoundingBox

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    #rects = cascade.detectMultiScale(img,minNeighbors=10, scaleFactor=1.05)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def print_image(frame,boxes,tag):
    if tag == "perfil_derecho":
        if len(boxes)==0:
            res_img = frame
        else:
            _,x_max,_= frame.shape
            boxes[0][0] = x_max-boxes[0][0]
            boxes[0][2] = x_max-boxes[0][2]
            names = [tag]*len(boxes)
            res_img = BoundingBox.bounding_box(frame,boxes,names)
    else:
        if len(boxes)==0:
            res_img = frame
        else:
            names = [tag]*len(boxes)
            res_img = BoundingBox.bounding_box(frame,boxes,names)
    return res_img


def models_profile(gray,detect_perfil_face):
    boxes =  detect(gray,detect_perfil_face)
    if len(boxes)!=0:
        tag = 'perfil_izquierdo'
        return boxes,tag
    else:
        gray_flipped = cv2.flip(gray, 1)
        boxes =  detect(gray_flipped,detect_perfil_face)
        if len(boxes)!=0:
            tag = 'perfil_derecho'
            return boxes,tag
        else:
            boxes=[]
            tag=[]
            return boxes,tag