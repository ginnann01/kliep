# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


class KLIEP():
    
    def __init__(self):
        pass
    
    def demo(self): 
        '''
        データ生成
        '''
        d = 1
        
        #データ量
        n_de = 100
        n_nu = 100
        
        #μ
        mu_de = 1
        mu_nu = 1
        
        #σ
        sigma_de = 1/2.0
        sigma_nu = 1/8.0
        
        x_de = mu_de+sigma_de*np.random.randn(d, n_de)
        x_nu = mu_nu+sigma_nu*np.random.randn(d, n_nu)
        
        xdisp = np.linspace(-0.5, 3, 100)
        p_de_xdisp = self.pdf_Gaussian(xdisp, mu_de, sigma_de)
        print p_de_xdisp[:,0:10]
        
    def pdf_Gaussian(self, x, mu, sigma):
        if len(np.shape(x)) == 1:
            d = 1
            nx = np.shape(x)[0]
        else:
            d = np.shape(x)[0]
            nx = np.shape(x)[1]
        
        tmp = ( x-np.tile(mu, [1, nx]) ) / np.tile(sigma, [1, nx])/np.sqrt(2)
        
        A = np.power( (2*np.pi), (-d/2.0) )
        B = (np.prod(sigma)*np.exp(-np.sum( np.power(tmp, 2), axis=0 )))
        print np.asarray([B]).shape
        print np.asarray([A]).shape
        return np.linalg.solve([[B]], [[A]])
        

if __name__=='__main__':
    k = KLIEP()
    k.demo()
