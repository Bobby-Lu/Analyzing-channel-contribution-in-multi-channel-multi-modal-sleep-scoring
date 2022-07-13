import numpy as np
na = np.newaxis

class Convolution():

    def __init__(self, activations, weights, bias, filtersize, stride):
        self.fh, self.fw, self.fd, self.n = filtersize
        self.stride = stride
        self.X = activations
        self.W = weights
        self.B = bias
    
    def lrp(self,R):
        N,NF,Hout,Wout = R.shape
        NF,df,hf,wf = self.W.shape
        hstride, wstride = self.stride
        Rx = np.zeros_like(self.X,dtype=np.float)
        self.W = np.where(self.W>0, self.W, 0)
        self.B = np.zeros_like(self.B,dtype=np.float)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.W[na,...] * self.X[:,na,:,i*hstride:i*hstride+hf , j*wstride:j*wstride+wf]
                Zs = Z.sum(axis=(2,3,4),keepdims=True) + self.B[na,...,na,na,na]
                Zs += 1e-16*((Zs >= 0)*2) 
                Rx[:, :,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf:  ] += ((Z/Zs) * R[:,:,na,i:i+1,j:j+1]).sum(axis=1)
        return Rx
    
class Convolution_first_layer():

    def __init__(self, activations, weights, bias, filtersize, stride):
        self.fh, self.fw, self.fd, self.n = filtersize
        self.stride = stride
        self.X = activations
        self.W = weights
        self.B = bias
    
    def lrp(self,R):
        N,NF,Hout,Wout = R.shape
        NF,df,hf,wf = self.W.shape
        hstride, wstride = self.stride
        Rx = np.zeros_like(self.X,dtype=np.float)
        self.B = np.zeros_like(self.B,dtype=np.float)
        for i in range(Hout):
            for j in range(Wout):
                Z = self.W[na,...] * self.X[:,na,:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf]
                Z = np.where(Z>0, Z, 0)
                Zs = Z.sum(axis=(2,3,4),keepdims=True) + self.B[na,...,na,na,na]
                Zs += 1e-16*((Zs >= 0)*2) 
                Rx[:, :,i*hstride:i*hstride+hf: , j*wstride:j*wstride+wf:  ] += ((Z/Zs) * R[:,:,na,i:i+1,j:j+1]).sum(axis=1)
        return Rx
   return R

    
class MaxPool():
    def __init__(self,activations,pool,stride):
        self.X = activations
        self.pool = pool
        self.stride = stride
    
    def lrp(self,R):
        N,D,H,W = self.X.shape
        hpool,   wpool   = self.pool
        hstride, wstride = self.stride
        Hout = (H - hpool) // hstride + 1
        Wout = (W - wpool) // wstride + 1
        Rx = np.zeros_like(self.X,dtype=np.float)
        self.Y = np.zeros((N,D,Hout,Wout))
        for i in range(Hout):
            for j in range(Wout):
                self.Y[:,:,i,j] = self.X[:, :, i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool:  ].max(axis=(2,3))  
        for i in range(Hout):
            for j in range(Wout):
                Z = self.Y[:,:,i:i+1,j:j+1,] == self.X[:,:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool  ]
                Zs = Z.sum(axis=(2,3),keepdims=True,dtype=np.float) 
                Rx[:,:,i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool] += (Z / Zs) * R[:,:,i:i+1,j:j+1]
        return Rx
    
    
class Flatten():
    def __init__(self,activations):
        self.X = activations
        self.inputshape = self.X.shape
    def lrp(self,R):
        return np.reshape(R,self.inputshape)
    
    
class Linear():
    def __init__(self,inputD,outputD,activations,weights,bias):
        self.m = inputD
        self.n = outputD
        self.B = bias
        self.W = weights
        self.X = activations
    def lrp(self,R):
        self.W = np.where(self.W>0, self.W, 0)
        self.B = np.zeros_like(self.B,dtype=np.float)
        Z = self.W[na,:,:]*self.X[:,:,na] 
        Zs = Z.sum(axis=1)[:,na,:] +self.B[na,na,:]
        Zs += 1e-16*((Zs >= 0)*2)
        return ((Z / Zs) * R[:,na,:]).sum(axis=2)