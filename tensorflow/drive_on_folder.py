import tensorflow as tf
import scipy.misc
import model
from subprocess import call
import matplotlib.pyplot as plt

#https://github.com/stsievert/python-drawnow
#from drawnow import drawnow, figure

# Load tensorflow section
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

# Load Wheel image
img = scipy.misc.imread('steering_wheel_image.jpg',0)
rows,cols,colors = img.shape

# Angle from model prediction
degrees = 0
# Variable used to filter prediction
smoothed_angle = 0
# Index for test image on "driving_dataset" directory
i = 1

    

# Loop until Crtl-C
try:    
    plt.figure(1)
    
    while True:
        # Read an image (index i) from directory "driving_dataset"
        full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".png", mode="RGB")
                
        image = scipy.misc.imresize(full_image, [66, 200]) / 255.0
        
        # Get steering angle from tensorflow model (Also convert from rad to degree)
        degrees = model.y.eval(feed_dict={model.x: [image], model.dropout_prob: 1.0})[0][0] * 180
        
        # Filter results (Smooth results between current angle and prediction)
        # Maybe this filter could be learn if we add a RNN at the end of the model
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)                              
        
        # Clear console                      
        call("clear")
        print("Predicted steering angle: " + str(degrees) + " degrees")
        print("Predicted(Filtered) steering angle: " + str(smoothed_angle) + " degrees")
        
        # Plot the image
        plt.subplot(211)        
        plt.imshow(full_image)
                
        # Rotate the wheel accordingly
        dst = scipy.misc.imrotate(img,-degrees)
        
        # Plot the steering wheel
        plt.subplot(212)
        plt.imshow(dst)        
        i += 1
        
        # Draw        
        plt.draw()        
        
        # Wait a bit        
        plt.pause(0.03)

	# Save images for video
        #plt.savefig("./imDriving/file%02d.png" % i)
except KeyboardInterrupt:
    pass


