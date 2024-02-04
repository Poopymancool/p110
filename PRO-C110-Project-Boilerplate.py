# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while(True):
      
    # Capture the video frame by frame
    ret, frame = camera.read()
  
    # Display the resulting frame
    img = cv2.resize(frame,(224,224))

    test_image=np.array(img,dtype=np.float32)
    test_image=np.expand_dims(test_image, axis=0)

    normalised_image=test_image/255.0

    prediction=model.predict(normalised_image)
    print("Prediction:", prediction)
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
camera.release()

# Destroy all the windows
cv2.destroyAllWindows()
