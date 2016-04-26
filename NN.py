from ptyprocess.ptyprocess import FileNotFoundError

import tensorflow as tf
import numpy as np
import random

'''
Feature Vectors are Right Ascension and Declination (these specify celestial coordinates)
 
 To Do: Include Date(?) and Magnitude (must normalize between different bands) as features
 
Labels are 2-dimensional one-hot vectors. (1,0) for eventual false positive, and (0,1) for
 an object which still has a chance to impact Earth.
'''

# GLOBAL FILENAMES
removed_fn = "Removed.txt"
not_removed_fn = "NotRemoved.txt"
removed_folder = "removed/"
not_removed_folder = "risks/"


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

# Do k-fold cross-validation on our NN
x_cp = np.copy(x_batch)
y_cp = np.copy(y_batch)
random.seed()
k = 5
n = x_batch.shape[0]
nks = [[]]*k
obs_bins = [[]]*k
lbl_bins = [[]]*k
for i in range(k):
    nks[i] = n / k
    if i == k:
        nks[i] = nks[i] + (n % k)*(k-1)
    for j in range(nks[i]):
        p = random.randint(0,x_cp.shape[0]-1)
        obs_bins[i].append(x_cp[p].tolist())
        x_cp = np.delete(x_cp, p, 0)
        
        lbl_bins[i].append(y_cp[p].tolist())
        y_cp = np.delete(y_cp, p, 0)

x_bins = []
y_bins = []
for i in range(k):
    x_bins.append(np.asarray(obs_bins[i]))
    y_bins.append(np.asarray(lbl_bins[i]))

def train_net(xtrain, ytrain, xtest, ytest): # Assume 2-dimensional parameters

    num_inputs = xtrain.shape[1]
    num_outputs = ytrain.shape[1]
    x = tf.placeholder(tf.float32, [None, num_inputs])

    W = tf.Variable(tf.zeros([num_inputs, num_outputs]))
    b = tf.Variable(tf.zeros([num_outputs]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, num_outputs])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={x: xtrain, y_: ytrain})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # We need to run this on a test set of data. This just tells training error
    accuracy = sess.run(accuracy, feed_dict={x: xtest, y_: ytest})
    test_error = 1.0 - accuracy
    
    return test_error

CV_k = 0
for i in range(k):
    x_train = np.zeros((0,0))
    y_train = np.zeros((0,0))
    for j in range(k):
        if j == i:
            continue
        if x_train.shape[0] == 0:
            x_train = x_bins[j]
            y_train = y_bins[j]
        else:
            np.vstack((x_train,x_bins[j]))
            np.vstack((y_train,y_bins[j]))
    x_test = x_bins[i]
    y_test = y_bins[i]
    CV_k = CV_k + (float(nks[i]) / n) * train_net(x_train, y_train, x_test, y_test)
    
print "Estimated test error by %d-fold cross-validation: %f" % (k, CV_k)