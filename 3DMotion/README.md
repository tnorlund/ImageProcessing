

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
```

# Structure From Motion

You can determine the positions of points in the scene and the motion of the camera.

## Epipolar Geometry

- The optical centers of the two cameras, a point <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/>, and the image points <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9a664305a06d03d8798d8bda4e14f517.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/> and <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/f467c5bf149d12e7bb28489e6164ebf9.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/> of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> all lie in the same plane (epipolar plane)
- These vectors are co-planar: <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/97a9f1e99243058ba72d4db884300a1e.svg?invert_in_darkmode" align=middle width=121.57310714999997pt height=31.799054100000024pt/>

- Now, instead of treating <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9a664305a06d03d8798d8bda4e14f517.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/> as a point, treat it as a 3D direction vector:

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/2c3d7c4da56904f3ae906f7f40b89c39.svg?invert_in_darkmode" align=middle width=104.96224695pt height=59.1786591pt/></p>
Here, we assume "normalized image coordinates (focal length of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>)

- The same can be applied in the second photo's coordinate system:
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/dd53607a8169baf04f23e42e8a7ea592.svg?invert_in_darkmode" align=middle width=104.96224695pt height=59.1786591pt/></p>
- To find the direction of the second photo's coordinate system in the first photo, you need to apply some rotation matrix <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> from the first photo's point, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9a664305a06d03d8798d8bda4e14f517.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/>, to the second photo's point, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/f467c5bf149d12e7bb28489e6164ebf9.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/>.
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/771efc4ad6882aa536f8b437dbb7a0c9.svg?invert_in_darkmode" align=middle width=44.09769375pt height=21.64371825pt/></p>
- Don't forget! These vectors determine direction and not the starting point.

- We know that the points <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9a664305a06d03d8798d8bda4e14f517.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/> and <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/f467c5bf149d12e7bb28489e6164ebf9.svg?invert_in_darkmode" align=middle width=14.823113249999992pt height=14.15524440000002pt/> are coplanar. This gives us the constraint
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/2d0734522faee69310e1ad790427dc08.svg?invert_in_darkmode" align=middle width=169.88326575pt height=19.09587735pt/></p>

- We can write the coplanar constraint as 
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/561783cea061531c051a88436e90a165.svg?invert_in_darkmode" align=middle width=124.72006635pt height=16.438356pt/></p>

  - Where <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> is the rotation of the first photo with respect to the second photo
  <p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9c5d0b0b77e5dfb75215726cdf5066d5.svg?invert_in_darkmode" align=middle width=29.274582149999997pt height=21.64371825pt/></p>
  - and <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> is the translation of the first photo's origin with respect to the first photo
  <p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/bec2ab3e3f57d737cb2350eeee343fc3.svg?invert_in_darkmode" align=middle width=59.6424213pt height=20.358039899999998pt/></p>
  - Remember that the pose, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/>, of the first photo with respect to the second photo is a homogeneous transformation matrix:
  <p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/c1a673a8899c1cce32616701f3e7205c.svg?invert_in_darkmode" align=middle width=184.78384319999998pt height=40.411106999999994pt/></p>
  
- We can replace the vector product with a skew symmetric-matrix for <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/c897b59e7d92116570986e2f517e5dc3.svg?invert_in_darkmode" align=middle width=95.62214309999999pt height=18.5680737pt/></p>
- We will let the essential matrix be the 3x3 matrix: <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/06b2af8e081df77c5dc613a9ecacd760.svg?invert_in_darkmode" align=middle width=70.95309374999998pt height=24.65753399999998pt/>.

- With all of this, we have the epipolar constraint:
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/a1a468d45c0b5e7e5000893e638b7171.svg?invert_in_darkmode" align=middle width=77.49020235pt height=18.7141317pt/></p>

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/0110a12f849f1a2933133f9ce1bce9cc.svg?invert_in_darkmode" align=middle width=309.3495537pt height=59.1786591pt/></p>




### Essential Matrix
- The matrix <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/>, that relates the image of a point in one camera to its image in the other camera, given a translation and rotation.
- With known coordinates, we can project/render images.



```python
image_size = 300
image_1 = np.zeros((image_size, image_size))
```

With a blank image, we can define pose, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/>. This will be composed of the focal length <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> and the translation to position 0.


```python
f = image_size
u_0 = image_size/2
v_0 = image_size/2
H = np.matrix([
    [f, 0, u_0],
    [0, f, v_0],
    [0, 0, 1]])
```

With the pose defined, we can define points on the cube.


```python
points = np.array([
    [0, 2, 0, 1], 
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [0, 2, -1, 1],
    [0, 1, -1, 1],
    [0, 0, -2, 1],
    [0, 2, -2, 1],
    [0, 1, -2, 1],
    [0, 0, -2, 2],
    [1, 0, 0, 1],
    [2, 0, 0, 1],
    [1, 0, -1, 1],
    [2, 0, -1, 1],
    [1, 0, -2, 1],
    [2, 0, -2, 1]
]).transpose()
```

With the points on the cube defined, we can rotate and translate the camera into its first position. We can do this by using an extrinsic camera parameter matrix, `M`. This matrix represents the camera rotated 120 degrees in the x coordinate system and 60 degrees in the z coordinate system. The camera is also translated 5 units in the z direction.


```python
a_x = 120 * (math.pi/180)
a_y = 0 * (math.pi/180)
a_z = 60 * (math.pi/180)
translate_0 = np.matrix([[0], [0], [5]])
R_x = np.matrix([
    [1, 0, 0],
    [0, math.cos(a_x), -math.sin(a_x)],
    [0, math.sin(a_x), math.cos(a_x)]
])
R_y = np.matrix([
    [math.cos(a_y), 0, math.sin(a_y)],
    [0, 1, 0],
    [-math.sin(a_y), 0, math.cos(a_y)]
])
R_z = np.matrix([
    [math.cos(a_z), -math.sin(a_z), 0], 
    [math.sin(a_z), math.cos(a_z), 0], 
    [0, 0, 1]
])
M = np.append(R_x * R_y * R_z, translate_0, 1)
points_1 = M * points
```

With the first image's focal length, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/5872d29d239f95cc7a5f43cfdd14fdae.svg?invert_in_darkmode" align=middle width=14.60053319999999pt height=22.831056599999986pt/>, and the point's distance from the camera, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f5bc204bf6a3d5abde8570c52d51cb6.svg?invert_in_darkmode" align=middle width=17.77402769999999pt height=22.465723500000017pt/>, set, we can render the individual points, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode" align=middle width=12.92146679999999pt height=14.15524440000002pt/>.

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/87c374b561c1d50f0be630f8db8d105e.svg?invert_in_darkmode" align=middle width=106.87697294999998pt height=16.438356pt/></p>

We then need to convert the normalized image points to be unnormalized.


```python
points_1[0,:] = points_1[0,:] / points_1[2,:]
points_1[1,:] = points_1[1,:] / points_1[2,:]
points_1[2,:] = points_1[2,:] / points_1[2,:]
u_1 = H * points_1
for row in u_1.transpose():
    image_1[int(row[0, 0])-2:int(row[0, 0])+2, int(row[0, 1])-2:int(row[0, 1])+2] = 255
plt.figure(1, figsize = (7.5,7.5))
axes = plt.gca()
axes.scatter(
    [x[0] for x in u_1.transpose()[:,0].tolist()],
    [300-y[0] for y in u_1.transpose()[:,1].tolist()]
)
axes.set_xlim([0, 300])
axes.set_ylim([0, 300])
axes.set_title("First Image", size=20)
axes.axis("off")
plt.show()
```


![png](html/output_10_0.png)


With the first image rendered, we can use the same calculations to move the camera again.


```python
a_x = 0 * (math.pi/180)
a_y = -25 * (math.pi/180)
a_z = 0 * (math.pi/180)
translate_1 = np.matrix([[3], [0], [1]])
R_x = np.matrix([
    [1, 0, 0],
    [0, math.cos(a_x), -math.sin(a_x)],
    [0, math.sin(a_x), math.cos(a_x)]
])
R_y = np.matrix([
    [math.cos(a_y), 0, math.sin(a_y)],
    [0, 1, 0],
    [-math.sin(a_y), 0, math.cos(a_y)]
])
R_z = np.matrix([
    [math.cos(a_z), -math.sin(a_z), 0], 
    [math.sin(a_z), math.cos(a_z), 0], 
    [0, 0, 1]
])
M_c1 = np.append(R_x * R_y * R_z, translate_1, 1)
```

With the extrinsic matrix created for the translation of camera 1 to camera 2, the different poses can be found. The pose of the start of the camera with respect to the camera's first position is created by appending the a row for rotation. The same is applied to the pose of the second position to the first position.


```python
H_m_c1 = np.vstack([M, [0, 0, 0, 1]])
H_c2_c1 = np.vstack([M_c1, [0, 0, 0, 1]])
H_c1_c2 = np.linalg.inv(H_c2_c1)
H_m_c2 = H_c1_c2 * H_m_c1
```


```python
R_m_c2 = H_m_c2[0:3,0:3]
translate_0_2 = H_m_c2[0:3,3]
M = np.append(R_m_c2, translate_0_2, 1)
```


```python
image_2 = np.zeros((image_size, image_size))
points_2 = np.vstack([
    (M * points)[0,:] / (M * points)[2,:],
    (M * points)[1,:] / (M * points)[2,:],
    (M * points)[2,:] / (M * points)[2,:]
])
u_2 = H * points_2
for row in u_2.transpose():
    image_2[int(row[0, 0])-2:int(row[0, 0])+2, int(row[0, 1])-2:int(row[0, 1])+2] = 255
    
plt.figure(1, figsize = (7.5,7.5))
axes = plt.gca()
axes.scatter(
    [x[0] for x in u_2.transpose()[:,0].tolist()],
    [300-y[0] for y in u_2.transpose()[:,1].tolist()]
)
axes.set_xlim([0, 300])
axes.set_ylim([0, 300])
axes.set_title("Second Image", size=20)
axes.axis("off")
plt.show()
```


![png](html/output_16_0.png)



```python
E = np.matrix([
    [0, -float(translate_1[2][0]), float(translate_1[1][0])],
    [float(translate_1[2][0]), 0, float(-translate_1[0][0])],
    [-float(translate_1[1][0]), float(translate_1[0][0]), 0]
]) * M_c1
E = np.delete(E, -1, axis=1)
E
```




    matrix([[ 0.        , -1.        ,  0.        ],
            [-0.361547  ,  0.        , -3.14154162],
            [ 0.        ,  3.        ,  0.        ]])



### Calculating the Essential Matrix

Remember the Essential matrix, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/>? It relates the image of a point in one image to its location in another image by a given rotation and/or translation,
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/8922441c3141b94c2b76a3096c545e1c.svg?invert_in_darkmode" align=middle width=82.05642719999999pt height=18.7141317pt/></p>
where
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/37c47babf473ee622155201cec619efb.svg?invert_in_darkmode" align=middle width=75.51931365pt height=16.438356pt/></p>

We can calculate <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/> if we know the pose between the two images. We can also calculate <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/> from a set of known point correspondences.

Now, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/> is a 3x3 matrix with 9 unknowns. Again, there is a scale factor, which helps us out by letting us know that there are actually only 8 unknowns. With 8 equations, we can calculate <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/> from 8 or more related points between images. We can simplify the equation as follows:

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/0110a12f849f1a2933133f9ce1bce9cc.svg?invert_in_darkmode" align=middle width=309.3495537pt height=59.1786591pt/></p>

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/bddbaeae06ca40f0a759443c366dbb2b.svg?invert_in_darkmode" align=middle width=302.01678209999994pt height=59.1786591pt/></p>

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/b9946944e8bec603243bed9bddb92060.svg?invert_in_darkmode" align=middle width=620.07641685pt height=14.42921205pt/></p>

With this equation, we can write it as a matrix equation by separating the unknowns from the known variables.

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/265d224d6f17e16b2fd95c89044ce941.svg?invert_in_darkmode" align=middle width=435.71986755pt height=108.49422870000001pt/></p>

With this system of homogeneous equations, we can ignore the trivial solution, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/8436d02a042a1eec745015a5801fc1a0.svg?invert_in_darkmode" align=middle width=39.53182859999999pt height=21.18721440000001pt/>, and find a unique solution that gives the smallest error. This means that we need to minimize 
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/bdb8829c8dd20cd5f2e5d96d2406d60c.svg?invert_in_darkmode" align=middle width=95.8238193pt height=26.301595649999996pt/></p>
since we know that it *should* be 0.

We can use singular value decomposition, SVD, to solve for <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>.
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/22a9e00bad13c6f2519549dd557cdf0b.svg?invert_in_darkmode" align=middle width=84.10434945pt height=14.6502939pt/></p>

The solution of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> is the column of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.242037049999992pt height=22.465723500000017pt/> corresponding to the only null singular value of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/>. In this scenario, the rightmost column of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode" align=middle width=13.242037049999992pt height=22.465723500000017pt/> is the solution we are looking for.

---

The first step in this process is to scale and translate the image points so that the centroid of all the points is found at the origin. In order to do this, we first need to calculate the centroids of all the (<img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/>) coordinates found in the first image.


```python
x_n = points_1[0:2, :]
N = x_n.shape[1]
t = np.sum(x_n, axis=1) / N
```

With this we can center the points to the origin, find the distance from the centroid of all the centered points, `dc`, and then calculate a scale factor `s` that we can use to scale the points to have an average distance of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/71486f265f83bc1e3d2b6f67704bcc23.svg?invert_in_darkmode" align=middle width=21.91788224999999pt height=28.511366399999982pt/> from the origin.


```python
xnc = x_n - t * np.ones((1,N))
dc = np.sqrt(np.sum(np.power(xnc, 2).transpose(), axis=1))
d_avg = 1 / N * np.sum(dc)
s = np.sqrt(2) / d_avg
```

Now we can create the <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> matrix with the scale factor and the distance from the centroid we found earlier.


```python
t_1 = np.zeros((2, 2), float)
np.fill_diagonal(t_1, s)
t_1 = np.vstack([np.hstack([t_1, -s * t]), [0, 0, 1]])
points_1_scaled = t_1 * points_1
```

We can do the same thing with the second set of points we found earlier.


```python
x_n = points_2[0:2, :]
N = x_n.shape[1]
t = np.sum(x_n, axis=1) / N
xnc = x_n - t * np.ones((1,N))
dc = np.sqrt(np.sum(np.power(xnc, 2).transpose(), axis=1))
d_avg = 1 / N * np.sum(dc)
s = np.sqrt(2) / d_avg
t_2 = np.zeros((2, 2), float)
np.fill_diagonal(t_2, s)
t_2 = np.vstack([np.hstack([t_2, -s * t]), [0, 0, 1]])
points_2_scaled = t_2 * points_2
```

In order to do our SVD, we first need to create the <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> matrix:
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/8122da2b3713b94f005fb8669ff54962.svg?invert_in_darkmode" align=middle width=355.3203753pt height=19.726228499999998pt/></p>

Because we have 15 unique points, the A matrix will be 15x9.


```python
A = np.hstack([
    np.multiply(points_1_scaled[0,:].transpose(), points_2_scaled[0,:].transpose()),
    np.multiply(points_1_scaled[0,:].transpose(), points_2_scaled[1,:].transpose()),
    points_1_scaled[0,:].transpose(),
    np.multiply(points_1_scaled[1,:].transpose(), points_2_scaled[0,:].transpose()),
    np.multiply(points_1_scaled[1,:].transpose(), points_2_scaled[1,:].transpose()),
    points_1_scaled[1,:].transpose(),
    points_2_scaled[0,:].transpose(),
    points_2_scaled[1,:].transpose(),
    np.matrix(np.ones(15)).transpose()
])
```

With the <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> matrix created, we can calculate the SVD of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/>.


```python
[U, D, V] = np.linalg.svd(A)
x = V[8, :]
E_scaled = np.reshape(V[8, :].tolist(), (3,3))
```

Now, we can force rank=2 and equal Eigenvalues in our scale.


```python
[U, D, V] = np.linalg.svd(E_scaled)
E_scaled = U * np.diag([1, 1, 0]) * V.transpose()
```


```python
E_compute =  t_1.transpose() * E_scaled * t_2
E_compute
```




    matrix([[ 2.45762485e-02,  0.00000000e+00,  2.86903168e-03],
            [ 0.00000000e+00, -1.72134999e+01,  1.22562695e+00],
            [ 9.64260721e-04,  1.21033188e+00, -8.60648732e-02]])



### Recovering Motion From The Essential Matrix

With the essential matrix, you can recover relative motion between images. We know that the Essential matrix is composed of the translation and rotation:
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/37c47babf473ee622155201cec619efb.svg?invert_in_darkmode" align=middle width=75.51931365pt height=16.438356pt/></p>

How do you recover this information? Again, you can use the SVD of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/>, 
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/2a96d43c90285573d72ccea1c0a334c4.svg?invert_in_darkmode" align=middle width=90.245826pt height=17.8466442pt/></p>
to recover <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> and <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/>.

The translation, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, is either <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/17511d3d954d9ac47cfe4638bb76a5a1.svg?invert_in_darkmode" align=middle width=15.96281939999999pt height=14.15524440000002pt/> or <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/031b32aed12ad03782bfae5768c8412b.svg?invert_in_darkmode" align=middle width=28.74825359999999pt height=19.1781018pt/>, where <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/17511d3d954d9ac47cfe4638bb76a5a1.svg?invert_in_darkmode" align=middle width=15.96281939999999pt height=14.15524440000002pt/> is the last column of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode" align=middle width=13.01596064999999pt height=22.465723500000017pt/>. The rotation, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> is either <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/903f724b34daa0198214a84be801d552.svg?invert_in_darkmode" align=middle width=53.599946399999986pt height=27.6567522pt/> or <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/21be078bca8075741fa35c292fa92e6b.svg?invert_in_darkmode" align=middle width=63.95553944999999pt height=27.6567522pt/>, where
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/f3cd54d7621f77319f19f4ffa482ebf9.svg?invert_in_darkmode" align=middle width=146.11869359999997pt height=59.1786591pt/></p>

This gives us 4 possible solutions. Luckily, 3 of the solutions support ideas that the points are behind the camera. We can rule these out. Our solution is homogeneous, which means that we can scale it by any constant, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, and get valid results. This gives us a correct rotation, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/>, but a translation, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, of an arbitrary amount.

---

First, we can calculate the SVD of <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.08219659999999pt height=22.465723500000017pt/> and define <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/>.


```python
[U, D, V] = np.linalg.svd(E)
W = np.matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
```

Now, we can calculate our 4 possible solutions.


```python
results = [
    np.vstack([np.hstack([U*W*V.transpose(),  U[:,2]]), [0,0,0,1]]),
    np.vstack([np.hstack([U*W*V.transpose(), -U[:,2]]), [0,0,0,1]]),
    np.vstack([np.hstack([U*W.transpose()*V.transpose(),  U[:,2]]), [0,0,0,1]]),
    np.vstack([np.hstack([U*W.transpose()*V.transpose(), -U[:,2]]), [0,0,0,1]])
]
# for result in results:
#     print(result)
results[3]
```




    matrix([[ 0.90630779,  0.        ,  0.42261826,  0.9486833 ],
            [ 0.        ,  1.        ,  0.        , -0.        ],
            [ 0.42261826,  0.        , -0.90630779,  0.31622777],
            [ 0.        ,  0.        ,  0.        ,  1.        ]])




```python
E
```




    matrix([[ 0.        , -1.        ,  0.        ],
            [-0.361547  ,  0.        , -3.14154162],
            [ 0.        ,  3.        ,  0.        ]])




```python
U
```




    matrix([[ 0.        , -0.31622777, -0.9486833 ],
            [ 1.        ,  0.        ,  0.        ],
            [ 0.        ,  0.9486833 , -0.31622777]])



### Epipolar Lines

You can draw any line on a 2D plane with the equation <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/9d7cfa8340a037cf6eb28353bedbd6c3.svg?invert_in_darkmode" align=middle width=111.22117214999999pt height=22.831056599999986pt/>. The issue we face while trying to draw lines between these points, is that you can scale this equation by any constant, <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, and the constraint is still meet: <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/0c44681eca5dc1843fb6ed661adcd719.svg?invert_in_darkmode" align=middle width=176.8035588pt height=24.65753399999998pt/>.

So, if we say that a line may be represented as a set of homogeneous coordinates:
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/23a4fde7e5b39438a1e26c1540160473.svg?invert_in_darkmode" align=middle width=95.61036705pt height=18.7598829pt/></p>
we can say that the point <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.270567249999992pt height=14.15524440000002pt/> lies on the line <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> if and only if <img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/33563bf7a4215502ecde4d1c3b13b5a6.svg?invert_in_darkmode" align=middle width=53.991344549999994pt height=27.6567522pt/>

<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/21545463f270b0e44472e4d7a843f239.svg?invert_in_darkmode" align=middle width=271.69475685pt height=59.1786591pt/></p>

With that out of the way, the constraint of the epipolar line, 
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/ff549eb34c0242f3a64f28c4b2d27f0f.svg?invert_in_darkmode" align=middle width=82.05642719999999pt height=18.7141317pt/></p>
allows us to draw lines on the images that are on the epipolar plane
<p align="center"><img src="https://rawgit.com/in	git@github.com:tnorlund/ImageProcessing/master/svgs/771c5587ef8079c11e43dbaa9383feab.svg?invert_in_darkmode" align=middle width=280.42793789999996pt height=59.1786591pt/></p>


```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))
axes[0].scatter(
    [x[0] for x in u_1.transpose()[:,0].tolist()],
    [300-y[0] for y in u_1.transpose()[:,1].tolist()])
axes[0].set_xlim([0, 300])
axes[0].set_ylim([0, 300])
axes[0].set_title("First Image", size=20)
axes[0].axis("off")
axes[1].scatter(
    [x[0] for x in u_2.transpose()[:,0].tolist()],
    [300-y[0] for y in u_2.transpose()[:,1].tolist()])
axes[1].set_xlim([0, 300])
axes[1].set_ylim([0, 300])
axes[1].set_title("Second Image", size=20)
axes[1].axis("off")
plt.show()
```


![png](html/output_40_0.png)



```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))
axes[0].scatter(
    [x[0] for x in u_1.transpose()[:,0].tolist()],
    [300-y[0] for y in u_1.transpose()[:,1].tolist()])
axes[0].set_xlim([0, 300])
axes[0].set_ylim([0, 300])
axes[0].set_title("First Image", size=20)
axes[0].axis("off")
axes[1].scatter(
    [x[0] for x in u_2.transpose()[:,0].tolist()],
    [300-y[0] for y in u_2.transpose()[:,1].tolist()])
axes[1].set_xlim([0, 300])
axes[1].set_ylim([0, 300])
axes[1].set_title("Second Image", size=20)
axes[1].axis("off")
for index in range(0, len(points_1.transpose())):
    l = E * points_2[:,index]
    error = float(points_1[:,index].transpose() * E * points_2[:,index])
    p_line_0 = H * np.matrix([
        [-1],
        [float((-l[2]-l[0]*(-1))/l[1])],
        [1]
    ])
    p_line_1 = H * np.matrix([
        [1],
        [float((-l[2]-l[0])/l[1])],
        [1]
    ])
    axes[0].plot(
        (float(p_line_0[0]), float(p_line_1[0])),
        (300-float(p_line_0[1]), 300-float(p_line_1[1])),
        'ro-'
    )
plt.show()
```


![png](html/output_41_0.png)



```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))
axes[0].scatter(
    [x[0] for x in u_1.transpose()[:,0].tolist()],
    [300-y[0] for y in u_1.transpose()[:,1].tolist()])
axes[0].set_xlim([0, 300])
axes[0].set_ylim([0, 300])
axes[0].set_title("First Image", size=20)
axes[0].axis("off")
axes[1].scatter(
    [x[0] for x in u_2.transpose()[:,0].tolist()],
    [300-y[0] for y in u_2.transpose()[:,1].tolist()])
axes[1].set_xlim([0, 300])
axes[1].set_ylim([0, 300])
axes[1].set_title("Second Image", size=20)
axes[1].axis("off")
for index in range(0, len(points_2.transpose())):
    l = E.transpose() * points_1[:,index]
    p_line_0 = H * np.matrix([
        [-1],
        [float((-l[2]-l[0]*(-1))/l[1])],
        [1]
    ])
    p_line_1 = H * np.matrix([
        [1],
        [float((-l[2]-l[0])/l[1])],
        [1]
    ])
    axes[1].plot(
        (float(p_line_0[0]), float(p_line_1[0])),
        (300-float(p_line_0[1]), 300-float(p_line_1[1])),
        'ro-'
    )
plt.show()
```


![png](html/output_42_0.png)

