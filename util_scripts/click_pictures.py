# A small helper script to click pictures via webcam to be
# used in enrollment in FaceAcess server.

import cv2

if __name__ == '__main__':
    name = input("Enter your full name (e.g. John Smith): ")
    name.replace(' ', '_')

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Preview")
    img_counter = 0

    while True:
        ret, frame = camera.read()
        cv2.imshow("Preview", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing capture...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "./images/{}_{}.jpg".format(name, img_counter)
            cv2.imwrite(img_name, frame)
            print("{} - saved!".format(img_name))
            img_counter += 1

            
