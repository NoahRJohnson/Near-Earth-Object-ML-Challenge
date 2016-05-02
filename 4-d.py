'''import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

z = np.array([0,1,2,3,4,5,6,7,8,9,10])
radius = np.array([0,1,1.5,1,0,2,4,5,4,2,1])
temp = np.array([150,200,210,220,225,220,195,185,160,150,140])

angle = np.linspace(0,2*np.pi,20)
Z,ANG = np.meshgrid(z,angle)
T,ANG = np.meshgrid(temp,angle)
# transform them to cartesian system
X,Y = radius*np.cos(ANG),radius*np.sin(ANG)

def fun():
    return [1,0]

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(T/float(T.max())))
plt.show()
'''



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Spherical coordinates
phi = np.linspace(0, 2 * np.pi, 100)
psi = np.linspace(0, np.pi, 100)

X = 10 * np.outer(np.cos(phi), np.sin(psi))
Y = 10 * np.outer(np.sin(phi), np.sin(psi))
Z = 10 * np.outer(np.ones(np.size(phi)), np.cos(psi))
xlen = len(X)
ylen = len(Y)
zlen = len(Z)
print xlen, ylen, len(Z), X.shape
print X.shape, Y.shape, Z.shape

'''colortuple = ('r', 'b')
colors = np.empty((xlen,ylen,zlen), dtype=str)
for y in range(ylen):
    for x in range(xlen):
        for z in range(zlen):
            colors[x, y, z] = colortuple[(x + y + z) % len(colortuple)]
'''       
T = np.empty((xlen,ylen,zlen))
T[:,:,2] = 0.8
#wireframe_sphere = ax.plot_wireframe(X,Y,Z,rstride=4,cstride=4)

X1 = X[:,:len(X)/3]
Y1 = Y[:,:len(Y)/3]
Z1 = Z[:,:len(Z)/3]
X2 = X[:,len(X)/3:]
Y2 = Y[:,len(Y)/3:]
Z2 = Z[:,len(Z)/3:]
surf = ax.scatter(X1.flatten(), Y1.flatten(), Z1.flatten(), c='r',
                       linewidth=0, antialiased=False)
surf = ax.scatter(X2, Y2, Z2, c='y',
                       linewidth=0, antialiased=False)

plt.show()