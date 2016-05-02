from ptyprocess.ptyprocess import FileNotFoundError

import tensorflow as tf
import numpy as np
import random
from matplotlib.dates import hours
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

'''
Feature Vectors are Right Ascension and Declination (these specify celestial coordinates)
 
 To Do: Include Date(?) and Magnitude (must normalize between different bands) as features
 
Labels are 2-dimensional one-hot vectors. (1,0) for eventual false positive, and (0,1) for
 an object which still has a chance to impact Earth.
'''

# GLOBALS
removed_fn = "Removed.txt"
not_removed_fn = "NotRemoved.txt"
removed_folder = "removed/"
not_removed_folder = "risks/"

TRAIN_OR_TEST = 1



def extract_features_from_observations(names_fn, folder):
    coords = []
    cnt = 0

    with open(names_fn, 'r') as names_f:
        for NEO_name in names_f.readlines():
            if (len(NEO_name) > 1 and NEO_name[len(NEO_name)-1] == "\n"):
                NEO_name = NEO_name[:-1] # Remove ending newline
            NEO_name = folder + NEO_name + ".txt"
            try:
                with open(NEO_name, 'r') as observations_f:
                    for observation in observations_f.readlines():
                        RA_s = observation[32:43]
                        RA_l = RA_s.split() # split on space
                        if len(RA_l) != 3:
                            print "error reading RA field for an observation in %s: wrong number of digits" % NEO_name
                            continue
                        RA_h = RA_l[0]
                        try:
                            RA_h = int(RA_h)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading RA HH field from %s" % NEO_name
                            RA_h = 0
                            
                        RA_m = RA_l[1]
                        try:
                            RA_m = int(RA_m)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading RA MM field from %s" % NEO_name
                            RA_m = 0
                            
                        RA_ss = RA_l[2]
                        try:
                            RA_ss = float(RA_ss)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading RA SS field from %s" % NEO_name
                            RA_ss = 0
                               
                    
                             
                        # Now declination
                        dec_s = observation[44:55]
                        dec_l = dec_s.split() # split on space
                        if len(dec_l) != 3:
                            print "error reading dec field for an observation in %s: wrong number of digits" % NEO_name
                            continue
                        dec_d = dec_l[0]
                        try:
                            dec_d = int(dec_d)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading dec DD field from %s" % NEO_name
                            dec_d = 0
                            
                        dec_m = dec_l[1]
                        try:
                            dec_m = int(dec_m)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading dec MM field from %s" % NEO_name
                            dec_m = 0
                        
                        dec_ss = dec_l[2]
                        try:
                            dec_ss = float(dec_ss)
                        except (NameError, ValueError): #TODO: proper error handling
                            print "error reading dec SS field from %s" % NEO_name
                            dec_ss = 0
                        
                
                        coords.append(RA_h)
                        coords.append(RA_m)
                        coords.append(RA_ss)
                        coords.append(dec_d)
                        coords.append(dec_m)
                        coords.append(dec_ss)
                    
                        cnt = cnt + 1
                    
            except IOError, FileNotFoundError:
                print "could not open file: %s" % NEO_name
    
    return (coords, cnt)
    
    
    
    
removed_l, cnt = extract_features_from_observations(removed_fn, removed_folder)
removed_ar = np.reshape(np.asarray(removed_l), (cnt,6))
removed_labels = np.reshape(np.asarray([1,0]*cnt), (cnt,2))

not_removed_l, cnt = extract_features_from_observations(not_removed_fn, not_removed_folder)
not_removed_ar = np.reshape(np.asarray(not_removed_l), (cnt,6))
not_removed_labels = np.reshape(np.asarray([0,1]*cnt), (cnt,2))

x_batch = np.vstack((removed_ar,not_removed_ar))
y_batch = np.vstack((removed_labels,not_removed_labels))

print x_batch.shape, y_batch.shape

# Now train our net

num_inputs = x_batch.shape[1]
num_outputs = y_batch.shape[1]
FV = tf.placeholder(tf.float32, [None, num_inputs])

W = tf.Variable(tf.zeros([num_inputs, num_outputs]), name="weights")
b = tf.Variable(tf.zeros([num_outputs]), name="biases")

labels = tf.nn.softmax(tf.matmul(FV, W) + b)

sess = tf.Session()


if (TRAIN_OR_TEST == 0):
    labels_ = tf.placeholder(tf.float32, [None, num_outputs])

    cross_entropy = -tf.reduce_sum(labels_*tf.log(tf.clip_by_value(labels,1e-10,1.0)))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    # Add op to save variables (weight and bias).
    saver = tf.train.Saver()

    
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={FV: x_batch, labels_: y_batch})

    # Save the variables to disk.
    saver.save(sess, "model.ckpt")
else:
    saver = tf.train.Saver()
    saver.restore(sess, "model.ckpt")


# Now train the model based on all of the data
#W,b = train_net(x_batch, y_batch)

# Add op to restore all the variables.
#saver = tf.train.Saver()
  
#FV = tf.placeholder(tf.float32, [1, x_batch.shape[1]])
#prediction = tf.nn.softmax(tf.matmul(FV, W) + b)


# Restore variables from disk.
#saver.restore(sess, "model.ckpt")

# Now create a visualization of the 4-d data using matplotlib

# Spherical coordinates define our celestial sphere
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
r = 10 # We don't care really what r is

X = r * np.outer(np.cos(phi), np.sin(theta))
Y = r * np.outer(np.sin(phi), np.sin(theta))
Z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))
xlen = len(X)
ylen = len(Y)
zlen = len(Z)

# Define change of coords from spherical to celestial
def SphericalToCelestial(theta, phi):
    
    decl = (np.pi / 2.0) - theta # conversion formula
    decl_degs = np.rad2deg(phi)
    decl_mins = (decl_degs - int(decl_degs))*60
    decl_secs = ((decl_mins) - int(decl_mins))*60
    
    decl_degs = int(decl_degs)
    decl_mins = int(decl_mins)
    decl_secs = round(decl_secs, 2)
    
    RA_hours = np.rad2deg(phi) / 15
    RA_mins = (RA_hours - int(RA_hours))*60
    RA_secs = (RA_mins - int(RA_mins))*60
    
    RA_hours = int(RA_hours)
    RA_mins = int(RA_mins)
    RA_seconds = round(RA_secs,2)
    return (RA_hours, RA_mins, RA_seconds, decl_degs, decl_mins, decl_secs)
    
    
all_sphere_points = []
cnt = 0
# Get every point on the celestial sphere in celestial coords
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # Get our 3-D data point in rectangular coords
        x = X[i][j]
        y = Y[i][j]
        z = Z[i][j]
        
        # Convert from rectangular to spherical coordinates
        theta = np.arccos(z / (np.sqrt(x**2 + y**2 + z**2)))
        phi = np.arctan2(y,x) # Note the special function arctan2 to handle edge cases like x == 0
            
        # Generate celestial coordinates
        rh, rm, rs, dd, dm, ds = SphericalToCelestial(theta, phi)
        
        # Append to our list
        cnt = cnt + 1
        all_sphere_points.append(float(rh))
        all_sphere_points.append(float(rm))
        all_sphere_points.append(float(rs))
        all_sphere_points.append(float(dd))
        all_sphere_points.append(float(dm))
        all_sphere_points.append(float(ds))
        
# Reshape list into correct shape
all_sphere_points = np.reshape(np.asarray(all_sphere_points), (cnt,6))
    
# Use the trained neural net to predict the class of our FV inputs
w_out = sess.run(W)
print w_out
b_out = sess.run(b)
print b_out
w_out = tf.Print(W, [W])
#fv_out = tf.Print(FV, [FV], name='fv')
mat_res = tf.matmul(FV,W)
a = tf.Print(mat_res, [mat_res])
NN_predictions = sess.run(labels, feed_dict={FV: all_sphere_points})
print NN_predictions, NN_predictions.shape

        
# Generate separate data points for FPs and non-FPs, which we will plot with appropriate colors
X_FPs = []
Y_FPs = []
Z_FPs = []

X_nonFPs = []
Y_nonFPs = []
Z_nonFPs = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # Get our 3-D data point in rectangular coords
        x = X[i][j]
        y = Y[i][j]
        z = Z[i][j]
        
        # Append this data point on the 3-D celestial sphere, to the correct list (for correct color) based on NN prediction.
        if NN_predictions[i*X.shape[0] + j][0] == 1:
            X_FPs.append(x)
            Y_FPs.append(y)
            Z_FPs.append(z)
        elif NN_predictions[i*X.shape[0] + j][1] == 1:
            X_nonFPs.append(x)
            Y_nonFPs.append(y)
            Z_nonFPs.append(z)
        else:
            print "Third hot vector class not supported."

        
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot scatter plots. I wish we could use surface_plot here, but it doesn't support 3-d colormaps
surf1 = ax.scatter(X_FPs, Y_FPs, Z_FPs, c = 'b', linewidth=0, antialiased=False)
surf2 = ax.scatter(X_nonFPs, Y_nonFPs, Z_nonFPs, c = 'r', linewidth=0, antialiased=False)

plt.show()
plt.savefig("Celestial_Sphere_classifications.png")