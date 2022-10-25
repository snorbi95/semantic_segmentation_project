import numpy as np
from matplotlib import pyplot as plt

indices = np.zeros((256,256,3))
values = np.zeros((256,256,3))

for i in range(3):
    p1 = np.random.random((1,256,256,2))
    indices_p1 = np.argmax(p1, axis=-1)
    indices_p1[indices_p1 != 0] += i
    values_p1 = np.max(p1, axis=-1)
    indices_p1 = indices_p1[0,:,:]
    values_p1 = values_p1[0,:,:]
    indices[:,:,i] = indices_p1
    values[:,:,i] = values_p1

prediction = np.zeros((256,256))

for i in range(256):
    for j in range(256):
        if np.sum(indices[i,j]) == 0:
            prediction[i,j] = 0
        else:
            max_class = 0
            max_value = 0
            for k in range(3):
                if values[i,j,k] > max_value and indices[i,j,k] != 0:
                    max_class = indices[i,j,k]
                    max_value = values[i,j,k]
            prediction[i,j] = max_class

plt.imshow(prediction)
plt.show()


