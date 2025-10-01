import cv2

img = cv2.imread("assets/pictures/pic1.jpg") #path to image

if img is None: 
    print("Could not read the image.")
else: 
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 