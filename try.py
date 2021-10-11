import numpy as np

#a = np.array([1,2,3,4])
#with open("./try.npy","wb") as f:
#    np.save(f, a)

b = np.load("./runs/margin_non_pretrain_3z/margins.npy")
#b = np.load("./runs/margin_non_pretrain_min/margins.npy")
print ("b is ", b)
print (type(b))
