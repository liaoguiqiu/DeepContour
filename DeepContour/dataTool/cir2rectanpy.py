import cv2
import os
import numpy as np
read_dir =  "C:/Users/u0132260/Documents/Data/ATLAS-0001/2021_08_25 case 5/TIFF_PDGT3Q3C/"
save_dir =  "C:/Users/u0132260/Documents/Data/ATLAS-0001/2021_08_25 case 5/TIFF_PDGT3Q3C_rect/"

def   tranfer_frome_cir2rec(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

        polar_image = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return polar_image
def tranfer_frome_rec2cir2(color, padding_H =58):
        H,W_ini,_ = color.shape
        padding = np.zeros((padding_H,W_ini,3))
         
        color  = np.append(padding,color,axis=0)
        H,W,_ = color.shape
        value = np.sqrt(((H/4.2)**2.0)+((W/4.2)**2.0))
        color=cv2.rotate(color,rotateCode = 2) 
        #value = 200
        #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
        #                               200, cv2.WARP_INVERSE_MAP)
        circular = cv2.linearPolar(color,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)

        circular = circular.astype(np.uint8)
        #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return circular

if __name__ == '__main__':
    read_start = 1 # the first
    read_sequence = os.listdir(read_dir) # read all file name
    #seqence_Len = len(read_sequence)    # get all file number 
    #img_path = operatedir_video +  read_sequence[0]
    #test_image = cv2.imread(img_path)  #read the first one to get the image size
    
    for img_id in read_sequence: # operate every image
        
        img_path = read_dir + img_id # starting from 10
        video = cv2.imread(img_path)
        gray_cir  =   cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        gray_rectan = tranfer_frome_cir2rec (gray_cir)
        # crop = gray_rectan[0:350,:] 
        crop = gray_rectan
        cv2.imwrite( save_dir +img_id ,crop)
        print(img_id)