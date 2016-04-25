from ptyprocess.ptyprocess import FileNotFoundError

import tensorflow as tf
import numpy as np

'''
Feature Vectors are Right Ascension and Declination (these specify celestial coordinates)
 
 To Do: Include Date(?) and Magnitude (must normalize between different bands) as features
 
Labels are 2-dimensional one-hot vectors. (1,0) for eventual false positive, and (0,1) for
 an object which still has a chance to impact Earth.
'''

# GLOBAL FILENAMES
removed_fn = "Removed.txt"
not_removed_fn = "NotRemoved.txt"


def extract_features_from_observations(fn):
    coords = []
    cnt = 0

    with open(fn, 'r') as names_f:
        for NEO_name in names_f.readlines():
            NEO_name = NEO_name + ".txt"
            try:
                with open(NEO_name, 'r') as observations_f:
                    for observation in observations_f.readlines():
                        RA_s = observation[32:43]
                        RA_l = RA_s.split() # split on space
                        if len(RA_l) != 3:
                            print "error reading RA field for an observation in %s: wrong number of digits" % NEO_name
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
    
    
    
    
removed_l, cnt = extract_features_from_observations(removed_fn)
removed_ar = np.reshape(np.asarray(removed_l), (cnt,6))
removed_labels = np.reshape(np.asarray([1,0]*cnt), (cnt,2))

not_removed_l, cnt = extract_features_from_observations(not_removed_fn)
not_removed_ar = np.reshape(np.asarray(not_removed_l), (cnt,6))
not_removed_labels = np.reshape(np.asarray([0,1]*cnt), (cnt,2))

x_batch = np.vstack((removed_ar,not_removed_ar))
y_batch = np.vstack((removed_labels,not_removed_labels))







x = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# We need to run this on a test set of data. This just tells training error
print(sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch}))
