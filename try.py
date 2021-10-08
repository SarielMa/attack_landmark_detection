import numpy as np

#a = np.array([1,2,3,4])
#with open("./try.npy","wb") as f:
#    np.save(f, a)

b = np.load("./runs/margin/margins.npy")
print ("b is ", b)
print (type(b))
