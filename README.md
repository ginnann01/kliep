# -*- coding: utf-8 -*-
'''
Kullback-Leiblar importance estimation procedure

杉山先生のMatlab実装
    http://www.ms.k.u-tokyo.ac.jp/software.html#KLIEP
のPython移植版

'''

import matplotlib.pyplot as plt
import numpy as np
import sys

class KLIEP():
    
    def __init__(self):
        pass

    def run(self, x_de, x_nu, x_re, sigma_chosen=0, b=100, fold=5):
        
        #引数の妥当性チェック
        if len(np.shape(x_de)) == 1:
            d = 1
            n_de = np.shape(x_de)[0]
        else:
            d = np.shape(x_de)[0]
            n_de = np.shape(x_de)[1]
        
        if len(np.shape(x_nu)) == 1:
            d_nu = 1
            n_nu = np.shape(x_nu)[0]
        else:
            d_nu = np.shape(x_nu)[0]
            n_nu = np.shape(x_nu)[1]

        if d != d_nu:
            sys.stderr.write('x_de and x_nu must have the same dimension')
        
        if sigma_chosen < 0:
            sys.stderr.write('Gaussian width must be positive')
        
        print 'Run KLIEP'
        
        #Choosing Gaussian kernel center 'x_ce'
        #rand_index = np.random.permutation(n_nu)
        rand_index = np.loadtxt('rand_index.csv', delimiter=',', dtype=np.int32)-1
        b = np.min([b, n_nu])
        x_ce = x_nu[:, rand_index[:b]]
        
        if sigma_chosen == 0:
            sigma = 10
            score = float("-inf")
            
        for epsilon in np.arange(np.log10(sigma)-1, -5, -1):
            for iteration in range(1, 9+1): #matlabの for c=a:b はpythonの for c in range(a,b+1)
                sigma_new = sigma-np.power(10, epsilon)
                
                #cv_index = np.random.permutation(n_nu)
                cv_index = np.loadtxt('cv_index.csv', delimiter=',', dtype=np.int32)-1
                cv_split = np.floor(np.arange(0, n_nu)*fold/n_nu)+1
                score_new = 0
                
                X_de = self.kernel_Gaussian(x_de, x_ce, sigma_new)
                X_nu = self.kernel_Gaussian(x_nu, x_ce, sigma_new)
                mean_X_de = np.mean(X_de, axis=0).T
                
                for i in range(1, fold+1):
                    alpha_cv, _ = self.KLIEP_learning(mean_X_de, X_nu[cv_index[cv_split!=i], :])
                    wh_cv = np.dot(X_nu[cv_index[cv_split == i], :], alpha_cv)
                    score_new = score_new + np.mean(np.log(wh_cv))/fold
                
                if score_new-score < 0:
                    break
                
                score = score_new
                sigma = sigma_new
                print "score = {}, sigma = {}".format(score, sigma)
        
        sigma_chosen=sigma
        print "sigma = {}".format(sigma_chosen)
        
        X_de = self.kernel_Gaussian(x_de, x_ce, sigma_chosen)
        X_nu = self.kernel_Gaussian(x_nu, x_ce, sigma_chosen)
        mean_X_de = np.mean(X_de, axis=0).T
        alphah, _ = self.KLIEP_learning(mean_X_de, X_nu)
        wh_x_de = np.dot(X_de, alphah).T
        
        if x_re is None:
            wh_x_re = float('nan')
        else:
            X_Re=self.kernel_Gaussian(x_re.reshape(1, -1), x_ce.reshape(1, -1), sigma_chosen)
            wh_x_re = np.dot(X_Re, alphah).T
        
        return wh_x_de, wh_x_re
        
    
    def kernel_Gaussian(self, x, c, sigma):
        if len(np.shape(x)) == 1:
            d = 1
            nx = np.shape(x)[0]
        else:
            d = np.shape(x)[0]
            nx = np.shape(x)[1]
        
        if len(np.shape(c)) == 1:
            d = 1
            nc = np.shape(c)[0]
        else:
            d = np.shape(c)[0]
            nc = np.shape(c)[1]
        
        x2 = np.sum(np.power(x, 2), axis=0).reshape(1, -1)
        c2 = np.sum(np.power(c, 2), axis=0).reshape(1, -1)

        distance2 = np.tile(c2, [nx, 1]) + np.tile(x2.T, [1, nc]) -2.0*x.T*c
        
        return np.exp(-distance2/(2.0*np.power(sigma, 2.0)))

    def KLIEP_learning(self, mean_X_de, X_nu):
        if len(np.shape(X_nu)) == 1:
            n_nu = 1
            nc = np.shape(X_nu)[0]
        else:
            n_nu = np.shape(X_nu)[0]
            nc = np.shape(X_nu)[1]
        
        max_iteration = 100
        epsilon_list = np.power(10.0, np.arange(3, -4, -1))
        c = np.sum( np.power(mean_X_de, 2) )
        alpha = np.ones( (nc,1) )
        [alpha, X_nu_alpha, score] = self.KLIEP_projection(alpha, X_nu, mean_X_de, c)
        
        for epsilon in epsilon_list:
            for iteration in range(1, max_iteration+1):
                alpha_tmp = alpha + np.dot( np.dot(epsilon, X_nu.T), (1/X_nu_alpha) )
                
                [alpha_new, X_nu_alpha_new, score_new] = self.KLIEP_projection(alpha_tmp, X_nu, mean_X_de, c)
                if score_new-score <= 0:
                    break
                score = score_new
                alpha = alpha_new
                X_nu_alpha = X_nu_alpha_new
        
        return alpha, score
    
    def KLIEP_projection(self, alpha, Xte, b, c):
        alpha = alpha + np.dot( np.dot(b, (1-np.sum(b*alpha.reshape(-1))) ).reshape(-1, 1), np.linalg.pinv(c.reshape(1, -1), rcond=1e-20) )
        alpha[alpha < 0] = 0

        alpha = np.dot( alpha, np.linalg.pinv(np.sum(b*alpha.reshape(-1)).reshape(1, -1), rcond=1e-20) )
        Xte_alpha = np.dot(Xte, alpha)
        score = np.mean(np.log(Xte_alpha))
        
        return alpha, Xte_alpha, score
            
    def pdf_Gaussian(self, x, mu, sigma):
        if len(np.shape(x)) == 1:
            d = 1
            nx = np.shape(x)[0]
        else:
            d = np.shape(x)[0]
            nx = np.shape(x)[1]
        
        tmp = ( x-np.tile(mu, [1, nx]) ) / np.tile(sigma, [1, nx])/np.sqrt(2)
        
        return ( np.power( (2*np.pi), (-d/2.0) )/np.prod(sigma) ) * np.exp(-np.sum( np.power(tmp, 2), axis=0 ))


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
        
        #x_de = mu_de+sigma_de*np.random.randn(d, n_de)
        #x_nu = mu_nu+sigma_nu*np.random.randn(d, n_nu)
        x_de = np.loadtxt('x_de.csv', delimiter=',').reshape(1, -1)
        x_nu = np.loadtxt('x_nu.csv', delimiter=',').reshape(1, -1)       
        
        xdisp = np.linspace(-0.5, 3, 100)
        p_de_xdisp = self.pdf_Gaussian(xdisp, mu_de, sigma_de)
        p_nu_xdisp = self.pdf_Gaussian(xdisp, mu_nu, sigma_nu)
        w_xdisp = p_nu_xdisp/p_de_xdisp
        
        p_de_x_de = self.pdf_Gaussian(x_de, mu_de, sigma_de)
        p_nu_x_de = self.pdf_Gaussian(x_de, mu_nu, sigma_nu)
        w_x_de = p_nu_x_de/p_de_x_de
        
        [wh_x_de, wh_xdisp] = self.run(x_de, x_nu, xdisp)
        
        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdisp, p_de_xdisp, 'b-', label='$p_{de}$(x)')
        ax.plot(xdisp, p_nu_xdisp, 'k-', label='$p_{nu}$(x)')
        ax.legend(loc="upper right")
        ax.set_xlabel('x')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdisp, w_xdisp, 'r-', label='w(x)')
        ax.plot(xdisp, wh_xdisp.reshape(-1), 'g-', label='w-hat(x)')
        ax.plot(x_de.reshape(-1), wh_x_de.reshape(-1), 'b.', label='w-hat($x^{de}$)')
        ax.legend(loc="upper right")
        ax.set_xlabel('x')
        
        plt.show()


if __name__=='__main__':
    k = KLIEP()
    k.demo()
