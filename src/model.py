import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)   
from scipy.special import xlogy, xlog1py   

class simple_logr_scaler:
    ### uniform scaler

    def __init__(self, X): ### 
        self.X = X

    def fit(self): ### scaler fit X
        self.n, self.m = self.X.shape[0], self.X.shape[1]
        self.v_max = self.X.max(axis = 0)
        self.v_min = self.X.min(axis = 0)
        self.dist = self.v_max - self.v_min

        X_scaled = -1 + 2 * np.einsum('ij, j-> ij', self.X - self.v_min, 1 / self.dist)
        return X_scaled


    def transform(self, Z): ## 把别的input Z 也按 X scale成 [-1,1]
        Z_scaled = -1 + 2 * np.einsum('ij, j-> ij', Z - self.v_min, 1 / self.dist)
        return Z_scaled

    def inverse_transform(self, Z_scaled): ## find original Z from scaled Z
        Z = self.v_min + np.einsum('ij, j-> ij', (Z_scaled + 1) / 2, self.dist) 
        return Z

class log_R_solver:

    def __init__(self, X, Y, alpha = 0.01, epsilon = 1e-6, lambda1 = 0.01, lambda2 = 0.01):
        self.X = X
        self.Y = Y
        self.n, self.m = X.shape[0], X.shape[1]
        self.theta = np.zeros(self.m + 1)
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
        self.Xs = np.ones((self.n, self.m + 1))
        self.Xs[:, : self.m] = self.X
        self.likely_hood_prev = 0

    def fit(self):
        i_iter = 0
        while True:
            likely_hood = (np.log(1 / (1 + np.exp(-self.Xs @ self.theta)))).T @ self.Y + (np.log(1 - 1 / (1 + np.exp(-self.Xs @ self.theta)))).T @ (1 - self.Y)
            self.theta = self.theta + self.alpha / self.n * self.Xs.T @ (self.Y - 1 / (1 + np.exp(-self.Xs @ self.theta)))
            likely_diff = np.abs(likely_hood - self.likely_hood_prev)
            if likely_diff < self.epsilon:
                break
            i_iter += 1
            self.likely_hood_prev = likely_hood
            if i_iter % 1000 == 0:
                print(f"iteration No.{i_iter}, likely_hood = {likely_hood}")
        return True

    def s_lambda(self, z):  ## prox function for L1 norm
        res = np.sign(z) * np.maximum(np.abs(z) - self.lambda1 * self.alpha, 0)
        return res

    def safe_sigmoid(self, z): ## safe version of sigmoid func
        THRESHOLD = 20.0
        mask_neg = z < -THRESHOLD
        mask_pos = z > THRESHOLD
        mask_mid = ~(mask_neg | mask_pos)

        res = np.empty_like(z)
        if np.any(mask_pos):
            res[mask_pos] = 1.0 - np.exp(-z[mask_pos])

        if np.any(mask_neg):
            res[mask_neg] = np.exp(z[mask_neg])

        if np.any(mask_mid):
            z = np.exp(-z[mask_mid])
            res[mask_mid] = 1.0 / (1.0 + z)

        return res

    def fit_L1(self):
        i_iter = 0
        while True:
            likely_hood = (np.log(1 / (1 + np.exp(-self.Xs @ self.theta)))).T @ self.Y + (np.log(1 - 1 / (1 + np.exp(-self.Xs @ self.theta)))).T @ (1 - self.Y) - self.lambda1 * np.linalg.norm(self.theta, ord=1)
            v = self.theta + self.alpha / self.n * self.Xs.T @ (self.Y - 1 / (1 + np.exp(-self.Xs @ self.theta)))
            self.theta[:-1] = self.s_lambda(v[:-1])
            self.theta[-1] = v[-1]
            likely_diff = np.abs(likely_hood - self.likely_hood_prev)
            if likely_diff < self.epsilon:
                break
            i_iter += 1
            self.likely_hood_prev = likely_hood
            if i_iter % 1000 == 0:
                print(f"iteration No.{i_iter}, likely_hood = {likely_hood}")
        return True

    def fit_L2(self):
        i_iter = 0
        while True:
            likely_hood = (np.log(self.safe_sigmoid(self.Xs @ self.theta))).T @ self.Y + (np.log(1 - self.safe_sigmoid(self.Xs @ self.theta))).T @ (1 - self.Y) - self.lambda2 * np.linalg.norm(self.theta) /2
            likely_hood_base = (np.log(self.safe_sigmoid(self.Xs @ self.theta))).T @ self.Y + (np.log(1 - self.safe_sigmoid(self.Xs @ self.theta))).T @ (1 - self.Y)
            likely_hood_reg = -self.lambda2 * np.linalg.norm(self.theta) /2
            likely_hood_ratio = likely_hood_reg / (likely_hood_reg + likely_hood_base)
            self.theta  = self.theta + self.alpha / self.n * (self.Xs.T @ (self.Y - self.safe_sigmoid(self.Xs @ self.theta)) + self.lambda2 * self.theta)
            grad = self.alpha / self.n * (self.Xs.T @ (self.Y - self.safe_sigmoid(self.Xs @ self.theta)) + self.lambda2 * self.theta)
            likely_diff = np.abs(likely_hood - self.likely_hood_prev)
            if likely_diff < self.epsilon:
                break
            i_iter += 1
            self.likely_hood_prev = likely_hood
            if i_iter % 1000 == 0:
                print(f"iteration No.{i_iter}, likely_hood = {likely_hood}, likely_hood_ratio = {likely_hood_ratio}, norm(theta) = {np.linalg.norm(self.theta):.8f}, norm(grad) = {np.linalg.norm(grad):.8f}, grad/theta ratio = {(np.linalg.norm(grad) / np.linalg.norm(self.theta)):.6f}")
            if i_iter > 50000:
                break
        return True

    def print_weights(self):
        return self.theta
    
    def transfrom(self, Z):
        Zs = np.ones((Z.shape[0], Z.shape[1] + 1))
        Zs[:, :Z.shape[1]] = Z
        res = 1 / (1 + np.exp(-Zs @ self.theta))
        res = np.where(res > 0.5, 1, 0)
        return res

class NN_solver:

    def __init__(self, X, Y, alpha = 0.01, dl1 = 3, dl2 = 2, dl3 = 1, epsilon = 1e-6, n_batch_size = 16, momentum = 0.9, n_seed = 40, output_gap = 5000):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.alpha = alpha
        self.dl1 = dl1
        self.dl2 = dl2
        self.dl3 = dl3
        self.epsilon = epsilon
        self.momentum = momentum
        self.n_seed = n_seed
        self.n_batch_size = n_batch_size
        self.output_gap = output_gap

    @staticmethod
    def safe_sigmoid(z): ## safe version of sigmoid func
        THRESHOLD = 20.0
        mask_neg = z < -THRESHOLD
        mask_pos = z > THRESHOLD
        mask_mid = ~(mask_neg | mask_pos)

        res = np.empty_like(z)
        if np.any(mask_pos):
            res[mask_pos] = 1.0 - np.exp(-z[mask_pos])

        if np.any(mask_neg):
            res[mask_neg] = np.exp(z[mask_neg])

        if np.any(mask_mid):
            z = np.exp(-z[mask_mid])
            res[mask_mid] = 1.0 / (1.0 + z)

        return res

    @staticmethod
    def relu(z):
        return np.maximum(0,z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(z.dtype)

    def lr_cosine_annealing(self, i_epoch, target_epoch, min_lr = 1e-4):
        return min_lr + 0.5 * (self.alpha - min_lr) * (
            1 + np.cos(np.pi * i_epoch / target_epoch)
        )

    def lr_decay_by_epoch(self, i_epoch, decay_factor = 1.0):
        return self.alpha / (1 + decay_factor * i_epoch)

    def fit_batch_adjust_lr(self, simple_iter_limit = 50000):
        
        self.W1_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, self.m))
        vW1 = np.zeros((self.dl1, self.m))
        self.b1_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, 1))
        vb1 = np.zeros((self.dl1, 1))
        self.W2_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, self.dl1))
        vW2 = np.zeros((self.dl2, self.dl1))
        self.b2_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, 1))
        vb2 = np.zeros((self.dl2, 1))
        self.W3_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, self.dl2))
        vW3 = np.zeros((self.dl3, self.dl2))
        self.b3_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, 1))
        vb3 = np.zeros((self.dl3, 1))

        i_epoch = 0
        loss = 0
        loss_prev = 1
        
        while True:
            
            idx_arr = np.random.permutation(self.n)
            n_iter = int(self.n / self.n_batch_size)
            
            for i in range(n_iter):
                idx = idx_arr[i * self.n_batch_size : (i+1) * self.n_batch_size]
                X_c = self.X[idx, :].T
                y_c = self.Y[idx]
                
                ## ======= forward prop
                z1 = self.W1_c @ X_c + self.b1_c  ## (3,n_batch_size)
                a1 = self.relu(z1)
    
                z2 = self.W2_c @ a1 + self.b2_c   ## (2,n_batch_size)
                a2 = self.relu(z2)
    
                z3 = self.W3_c @ a2 + self.b3_c   ## (1,n_batch_size)
                a3 = self.safe_sigmoid(z3)
               
                ## ======= backward prop
                dLz3 = a3 - y_c   ## (1,n) 
                dLb3 = np.mean(dLz3, axis = 1, keepdims=True)  ## (1,1)
                ## W3 (1,2), dLW3: (1,2)
                dLW3 = 1/dLz3.shape[1] * dLz3 @ a2.T ## (3,2) 
    
                ## W3.T: (2,1)  
                dLz2 = (self.W3_c.T @ dLb3) * self.relu_derivative(z2)  ## (2,n_batch_size)
                dLb2 = np.mean(dLz2, axis = 1, keepdims=True)  ## (2,1)
                ## W2: (2,3), dLW2: (2,3)
                dLW2 = 1/dLz2.shape[1] * dLz2 @ a1.T 
    
                ## W2.T: (3,2)
                dLz1 = (self.W2_c.T @ dLb2) * self.relu_derivative(z1)  ## (3,n_batch_size)
                dLb1 = np.mean(dLz1, axis = 1, keepdims=True)  ## (3,1)
                ## W1: (3,m), dLW1: (3,m), X:(n, m)
                dLW1 = 1/dLz1.shape[1] * dLz1 @ X_c.T
    
                ## ======= parameter update
                vW3 = self.momentum * vW3 + (1-self.momentum) * dLW3
                self.W3_c = self.W3_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vW3
                vb3 = self.momentum * vb3 + (1-self.momentum) * dLb3
                self.b3_c = self.b3_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vb3
                vW2 = self.momentum * vW2 + (1-self.momentum) * dLW2
                self.W2_c = self.W2_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vW2
                vb2 = self.momentum * vb2 + (1-self.momentum) * dLb2
                self.b2_c = self.b2_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vb2
                vW1 = self.momentum * vW1 + (1-self.momentum) * dLW1
                self.W1_c = self.W1_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vW1
                vb1 = self.momentum * vb1 + (1-self.momentum) * dLb1
                self.b1_c = self.b1_c - self.lr_cosine_annealing(i_epoch, simple_iter_limit) * vb1

            
            i_epoch += 1

            z1_e = self.W1_c @ self.X.T + self.b1_c  ## (3,n)
            a1_e = self.relu(z1_e)

            z2_e = self.W2_c @ a1_e + self.b2_c   ## (2,n)
            a2_e = self.relu(z2_e)

            z3_e = self.W3_c @ a2_e + self.b3_c   ## (1,n)
            a3_e = self.safe_sigmoid(z3_e)


            #  loss = - np.sum(self.Y.T * np.log(a3_e) + (1-self.Y.T) * np.log(1-a3_e))
            loss = - np.sum(xlogy(a3_e, self.Y.T) + xlog1py(-a3_e, 1-self.Y.T))
            loss_diff = np.abs(loss - loss_prev)
            loss_prev = loss
            if i_epoch % self.output_gap == 1:
                print(f"epoch No.{i_epoch}, loss = {loss:.6f}")
            
            ## terminiate condition
            if i_epoch > simple_iter_limit * 10:
                break
            if loss_diff < self.epsilon:
                print("converged")
                break

        return True

    def fit(self, simple_iter_limit = 50000):
        
        self.W1_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, self.m))
        vW1 = np.zeros((self.dl1, self.m))
        self.b1_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, 1))
        vb1 = np.zeros((self.dl1, 1))
        self.W2_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, self.dl1))
        vW2 = np.zeros((self.dl2, self.dl1))
        self.b2_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, 1))
        vb2 = np.zeros((self.dl2, 1))
        self.W3_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, self.dl2))
        vW3 = np.zeros((self.dl3, self.dl2))
        self.b3_c = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, 1))
        vb3 = np.zeros((self.dl3, 1))

        vW1 = np.zeros(self.W1_c.shape)
        vb1 = np.zeros(self.b1_c.shape)
        vW2 = np.zeros(self.W2_c.shape)
        vb2 = np.zeros(self.b2_c.shape)
        vW3 = np.zeros(self.W3_c.shape)
        vb3 = np.zeros(self.b3_c.shape)

        i_iter = 0
        loss = 0
        loss_prev = 1
        while True:
            X_c = self.X.T
            y_c = self.Y.T
            
            ## ======= forward prop
            z1 = self.W1_c @ X_c + self.b1_c  ## (3,n)
            a1 = self.relu(z1)

            z2 = self.W2_c @ a1 + self.b2_c   ## (2,n)
            a2 = self.relu(z2)

            z3 = self.W3_c @ a2 + self.b3_c   ## (1,n)
            a3 = self.safe_sigmoid(z3)

            loss = - np.sum(y_c * np.log(a3) + (1-y_c) * np.log(1-a3))
            if i_iter % self.output_gap == 1:
                print(f"iter No.{i_iter}, loss = {loss:.6f}")
           
            ## ======= backward prop
            dLz3 = a3 - y_c   ## (1,n) 
            dLb3 = np.mean(dLz3, axis = 1, keepdims=True)  ## (1,1)
            ## W3 (1,2), dLW3: (1,2)
            dLW3 = 1/self.n * dLz3 @ a2.T ## (3,2) 

            ## W3.T: (2,1)  
            dLz2 = (self.W3_c.T @ dLb3) * self.relu_derivative(z2)  ## (2,n)
            dLb2 = np.mean(dLz2, axis = 1, keepdims=True)  ## (2,1)
            ## W2: (2,3), dLW2: (2,3)
            dLW2 = 1/self.n * dLz2 @ a1.T 

            ## W2.T: (3,2)
            dLz1 = (self.W2_c.T @ dLb2) * self.relu_derivative(z1)  ## (3,n)
            dLb1 = np.mean(dLz1, axis = 1, keepdims=True)  ## (3,1)
            ## W1: (3,m), dLW1: (3,m), X:(n, m)
            dLW1 = 1/self.n * dLz1 @ X_c.T

            ## ======= parameter update
            vW3 = self.momentum * vW3 + (1-self.momentum) * dLW3
            self.W3_c = self.W3_c - self.alpha * vW3
            vb3 = self.momentum * vb3 + (1-self.momentum) * dLb3
            self.b3_c = self.b3_c - self.alpha * vb3
            vW2 = self.momentum * vW2 + (1-self.momentum) * dLW2
            self.W2_c = self.W2_c - self.alpha * vW2
            vb2 = self.momentum * vb2 + (1-self.momentum) * dLb2
            self.b2_c = self.b2_c - self.alpha * vb2
            vW1 = self.momentum * vW1 + (1-self.momentum) * dLW1
            self.W1_c = self.W1_c - self.alpha * vW1
            vb1 = self.momentum * vb1 + (1-self.momentum) * dLb1
            self.b1_c = self.b1_c - self.alpha * vb1
            
            i_iter += 1
            loss_diff = np.abs(loss - loss_prev)
            
            ## terminiate condition
            if i_iter > simple_iter_limit:
                break
            if loss_diff < self.epsilon:
                break

        return True

    def fit_shallow_parallel(self, shallow_iter_limit = 2000, target_loss = 400):
        ## define and initialize the computing blocks

        self.W1 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, self.m, self.n_seed))
        vW1 = np.zeros(self.W1.shape)
        self.b1 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, 1, self.n_seed))
        vb1 = np.zeros(self.b1.shape)
        self.W2 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, self.dl1, self.n_seed))
        vW2 = np.zeros(self.W2.shape)
        self.b2 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, 1, self.n_seed))
        vb2 = np.zeros(self.b2.shape)
        self.W3 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, self.dl2, self.n_seed))
        vW3 = np.zeros(self.W3.shape)
        self.b3 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, 1, self.n_seed))
        vb3 = np.zeros(self.b3.shape)

        self.loss_arr = np.zeros(self.n_seed)
        self.loss_arr_prev = np.zeros(self.n_seed)

        i_epoch = 0
        i_loc = 0
        loss = 0
        loss_prev = 1
        
        while True:
            idx_arr = np.random.permutation(self.n)
            n_iter = int(self.n / self.n_batch_size)

            for j in range(n_iter):
                idx = idx_arr[j * self.n_batch_size : (j+1) * self.n_batch_size]
                X_c = self.X[idx, :].T
                y_c = self.Y[idx]  ## (n_batch_size)

                ## ======= forward prop
                ## W1(3,m,n_seed), X_c = X[idx].T (m, n_batch_size), b1 (3,1,n_seed)
                z1 = np.einsum('ijk, jl -> ilk', self.W1, X_c) + self.b1   ## (3,n_batch_size, n_seed)
                a1 = self.relu(z1)

                ## W2(2,3,n_seed), a1(3,n_batch_size,n_seed), b2(2,1,n_seed)
                z2 = np.einsum('ijk, jlk -> ilk', self.W2, a1) + self.b2   ## (2,n_batch_size, n_seed)
                a2 = self.relu(z2)

                ## W3(1,2,n_seed), a2(2,n_batch_size,n_seed), b3(1,1,n_seed)
                z3 = np.einsum('ijk, jlk -> ilk', self.W3, a2) + self.b3   ## (1,n_batch_size, n_seed)
                a3 = self.safe_sigmoid(z3)
           
                ## ======= backward prop
                dLz3 = a3 - y_c[np.newaxis,:,np.newaxis]   ## (1,n_batch_size, n_seed)
                dLb3 = np.mean(dLz3, axis = 1, keepdims=True)  ## (1,1, n_seed)
                ## W3 (1,2, n_seed), dLz3: (1,n_batch_size, n_seed), a2 (2,n_batch_size, n_seed)
                dLW3 = 1/dLz3.shape[1] * np.einsum('ilk, jlk-> ijk', dLz3, a2) ## (1,2, n_seed) 

                ## W3 (1,2, n_seed), dLz3: (1,n_batch_size, n_seed), z2 (2,n_batch_size, n_seed)
                dLz2 = np.einsum('ijk, ilk -> jlk', self.W3, dLz3) * self.relu_derivative(z2) ## (2,n_batch_size, n_seed)
                dLb2 = np.mean(dLz2, axis = 1, keepdims=True)  ## (2,1, n_seed)
                ## W2: (2,3, n_seed), dLz2: (2,n_batch_size, n_seed), a1 (3,n_batch_size, n_seed)
                dLW2 = 1/dLz2.shape[1] * np.einsum('ilk, jlk-> ijk', dLz2, a1) ## (2,3, n_seed) 

                ## W2 (2,3, n_seed), dLz2: (2,n_batch_size, n_seed), z1 (3,n_batch_size, n_seed)
                dLz1 = np.einsum('ijk, ilk -> jlk', self.W2, dLz2) * self.relu_derivative(z1) ## (3,n_batch_size, n_seed)
                dLb1 = np.mean(dLz1, axis = 1, keepdims=True)  ## (3,1, n_seed)
                ## W1: (3,m, n_seed), dLz1: (3,n_batch_size, n_seed), X_c:(m, n_batch_size)
                dLW1 = 1/dLz1.shape[1] * np.einsum('ilk, jl-> ijk', dLz1, X_c) ## (3,m, n_seed) 


                vW3 = self.momentum * vW3 + (1-self.momentum) * dLW3
                self.W3 = self.W3 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW3

                vb3 = self.momentum * vb3 + (1-self.momentum) * dLb3
                self.b3 = self.b3 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb3
                
                vW2 = self.momentum * vW2 + (1-self.momentum) * dLW2
                self.W2 = self.W2 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW2
                
                vb2 = self.momentum * vb2 + (1-self.momentum) * dLb2
                self.b2 = self.b2 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb2
                
                vW1 = self.momentum * vW1 + (1-self.momentum) * dLW1
                self.W1 = self.W1 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW1
                
                vb1 = self.momentum * vb1 + (1-self.momentum) * dLb1
                self.b1 = self.b1 - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb1

            
            i_epoch += 1

            ### 这里 a, z 要根据 n 重新算一遍
            ## W1(3,m,n_seed), X (n,m), b1 (3,1,n_seed)
            z1_e = np.einsum('ijk, lj -> ilk', self.W1, self.X)  + self.b1  ## (3,n, n_seed)
            a1_e = self.relu(z1_e)

            ## W2(2,3,n_seed), a1_e (3,n, n_seed), b2 (2,1,n_seed)
            z2_e = np.einsum('ijk, jlk -> ilk', self.W2, a1_e)  + self.b2   ## (2,n, n_seed)
            a2_e = self.relu(z2_e)

            ## W3(1,2,n_seed), a2_e (2,n, n_seed), b3 (1,1,n_seed)
            z3_e = np.einsum('ijk, jlk -> ilk', self.W3, a2_e)  + self.b3   ## (1,n, n_seed)
            a3_e = self.safe_sigmoid(z3_e)

            ### loss_arr (n_seed), Y (n), a3_e (1,n, n_seed)
            # self.loss_arr = - np.sum(self.Y[np.newaxis, :, np.newaxis] * np.log(a3_e) + (1-self.Y[np.newaxis, :, np.newaxis]) * np.log(1-a3_e), axis=1) 
            self.loss_arr = - np.sum(xlogy(self.Y[np.newaxis, :, np.newaxis], a3_e) + xlog1py(1-self.Y[np.newaxis, :, np.newaxis], -a3_e), axis=1) 
            loss_diff = np.abs(self.loss_arr - self.loss_arr_prev)
            self.loss_arr_prev = self.loss_arr

            if i_epoch % self.output_gap == 1:
                print(f"epoch No.{i_epoch}, min loss = {self.loss_arr.min():.6f}, from seed No.{self.loss_arr.argmin()}")

            if i_epoch > shallow_iter_limit*2:
                break

        return True

    def fit_shallow(self, shallow_iter_limit = 2000, target_loss = 400):
        ## define and initialize the computing blocks

        self.W1 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, self.m, self.n_seed))
        vW1 = np.zeros((self.dl1, self.m))
        self.b1 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl1 + self.m)), size = (self.dl1, 1, self.n_seed))
        vb1 = np.zeros((self.dl1, 1))
        self.W2 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, self.dl1, self.n_seed))
        vW2 = np.zeros((self.dl2, self.dl1))
        self.b2 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl2 + self.dl1)), size = (self.dl2, 1, self.n_seed))
        vb2 = np.zeros((self.dl2, 1))
        self.W3 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, self.dl2, self.n_seed))
        vW3 = np.zeros((self.dl3, self.dl2))
        self.b3 = np.random.normal(loc = 0, scale = np.sqrt(2 / (self.dl3 + self.dl2)), size = (self.dl3, 1, self.n_seed))
        vb3 = np.zeros((self.dl3, 1))

        self.loss_arr = np.ones(self.n_seed) * 10000

        for i in range(self.n_seed):         
            i_epoch = 0
            i_loc = 0
            loss = 0
            loss_prev = 1
            target_met = False
            
            while True:
                idx_arr = np.random.permutation(self.n)
                n_iter = int(self.n / self.n_batch_size)

                for j in range(n_iter):
                    idx = idx_arr[j * self.n_batch_size : (j+1) * self.n_batch_size]
                    X_c = self.X[idx, :].T
                    y_c = self.Y[idx]

                    ## ======= forward prop
                    ## W1(3,m), X_c = X[idx].T (m, n_batch)
                    z1 = self.W1[:, :, i] @ X_c + self.b1[:, :, i]  ## (3,n_batch_size)
                    a1 = self.relu(z1)
    
                    z2 = self.W2[:, :, i] @ a1 + self.b2[:, :, i]   ## (2,n_batch_size)
                    a2 = self.relu(z2)
        
                    z3 = self.W3[:, :, i] @ a2 + self.b3[:, :, i]   ## (1,n_batch_size)
                    a3 = self.safe_sigmoid(z3)
               
                    ## ======= backward prop
                    dLz3 = a3 - y_c   ## (1,n_batch) 
                    dLb3 = np.mean(dLz3, axis = 1, keepdims=True)  ## (1,1)
                    ## W3 (1,2), dLW3: (1,2)
                    dLW3 = 1/dLz3.shape[1] * dLz3 @ a2.T ## (3,2) 
    
                    ## W3.T: (2,1)  
                    dLz2 = (self.W3[:, :, i].T @ dLb3) * self.relu_derivative(z2)  ## (2,n_batch_size)
                    dLb2 = np.mean(dLz2, axis = 1, keepdims=True)  ## (2,1)
                    ## W2: (2,3), dLW2: (2,3)
                    dLW2 = 1/dLz2.shape[1] * dLz2 @ a1.T 
    
                    ## W2.T: (3,2)
                    dLz1 = (self.W2[:, :, i].T @ dLb2) * self.relu_derivative(z1)  ## (3,n_batch_size)
                    dLb1 = np.mean(dLz1, axis = 1, keepdims=True)  ## (3,1)
                    ## W1: (3,m), dLW1: (3,m), X:(n, m)
                    dLW1 = 1/dLz1.shape[1] * dLz1 @ X_c.T


                    vW3 = self.momentum * vW3 + (1-self.momentum) * dLW3
                    self.W3[:, :, i] = self.W3[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW3
    
                    vb3 = self.momentum * vb3 + (1-self.momentum) * dLb3
                    self.b3[:, :, i] = self.b3[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb3
                    
                    vW2 = self.momentum * vW2 + (1-self.momentum) * dLW2
                    self.W2[:, :, i] = self.W2[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW2
                    
                    vb2 = self.momentum * vb2 + (1-self.momentum) * dLb2
                    self.b2[:, :, i] = self.b2[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb2
                    
                    vW1 = self.momentum * vW1 + (1-self.momentum) * dLW1
                    self.W1[:, :, i] = self.W1[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vW1
                    
                    vb1 = self.momentum * vb1 + (1-self.momentum) * dLb1
                    self.b1[:, :, i] = self.b1[:, :, i] - self.lr_cosine_annealing(i_epoch, shallow_iter_limit) * vb1

                
                i_epoch += 1

                ### 这里 a, z 要根据 n 重新算一遍
                z1_e = self.W1[:, :, i] @ self.X.T + self.b1[:, :, i]  ## (3,n)
                a1_e = self.relu(z1_e)
    
                z2_e = self.W2[:, :, i] @ a1_e + self.b2[:, :, i]   ## (2,n)
                a2_e = self.relu(z2_e)
    
                z3_e = self.W3[:, :, i] @ a2_e + self.b3[:, :, i]   ## (1,n)
                a3_e = self.safe_sigmoid(z3_e)
                
                loss = - np.sum(self.Y.T * np.log(a3_e) + (1-self.Y.T) * np.log(1-a3_e))
                loss_diff = np.abs(loss - loss_prev)
                loss_prev = loss
                self.loss_arr[i] = loss

                if i_epoch % self.output_gap == 1:
                    print(f"epoch No.{i_epoch}, loss = {loss:.6f}")

                if i_epoch > shallow_iter_limit*2:
                    break
                
                if loss < target_loss:
                    print(f"Seed.{i} epoch No.{i_epoch}, loss = {loss:.6f}")
                    target_met = True
                    break

                if loss_diff < self.epsilon:
                    print("converged")
                    break

            print(f"Seed.{i}, loss = {loss:.6f}")
            print(f"Current min loss = {self.loss_arr.min()}, from seed {self.loss_arr.argmin()}")
            self.min_loss = self.loss_arr.min()
            self.min_idx = self.loss_arr.argmin()
            if target_met:
                break

        return self.min_loss, self.min_idx

    def fit_deep(self, deep_iter_limit = 10000, lr_raito = 1, use_input = False, W1_i=None, b1_i=None, W2_i=None, b2_i=None, W3_i=None, b3_i=None):
        
        if use_input:
            self.W1_c = W1_i
            self.b1_c = b1_i
            self.W2_c = W2_i
            self.b2_c = b2_i
            self.W3_c = W3_i
            self.b3_c = b3_i
        else:
            self.W1_c = self.W1[:, :, self.min_idx]
            self.b1_c = self.b1[:, :, self.min_idx]
            self.W2_c = self.W2[:, :, self.min_idx]
            self.b2_c = self.b2[:, :, self.min_idx]
            self.W3_c = self.W3[:, :, self.min_idx]
            self.b3_c = self.b3[:, :, self.min_idx]

        vW1 = np.zeros(self.W1_c.shape)
        vb1 = np.zeros(self.b1_c.shape)
        vW2 = np.zeros(self.W2_c.shape)
        vb2 = np.zeros(self.b2_c.shape)
        vW3 = np.zeros(self.W3_c.shape)
        vb3 = np.zeros(self.b3_c.shape)

        i_epoch = 0
        loss = 0
        loss_prev = 1
        while True:
            idx_arr = np.random.permutation(self.n)
            n_iter = int(self.n / self.n_batch_size)

            for i in range(n_iter):
                idx = idx_arr[i * self.n_batch_size : (i+1) * self.n_batch_size]
                X_c = self.X[idx, :].T
                y_c = self.Y[idx].T
                
                ## ======= forward prop
                z1 = self.W1_c @ X_c + self.b1_c  ## (3,n)
                a1 = self.relu(z1)
    
                z2 = self.W2_c @ a1 + self.b2_c   ## (2,n)
                a2 = self.relu(z2)
    
                z3 = self.W3_c @ a2 + self.b3_c   ## (1,n)
                a3 = self.safe_sigmoid(z3)
               
                ## ======= backward prop
                dLz3 = a3 - y_c   ## (1,n_batch_size)
                dLb3 = np.mean(dLz3, axis = 1, keepdims=True)  ## (1,1)
                ## W3 (1,2), dLW3: (1,2)
                dLW3 = 1/dLz3.shape[1] * dLz3 @ a2.T ## (3,2) 
    
                ## W3.T: (2,1)  
                dLz2 = (self.W3_c.T @ dLb3) * self.relu_derivative(z2)  ## (2,n_batch_size)
                dLb2 = np.mean(dLz2, axis = 1, keepdims=True)  ## (2,1)
                ## W2: (2,3), dLW2: (2,3)
                dLW2 = 1/dLz2.shape[1] * dLz2 @ a1.T 
    
                ## W2.T: (3,2)
                dLz1 = (self.W2_c.T @ dLb2) * self.relu_derivative(z1)  ## (3,n_batch_size)
                dLb1 = np.mean(dLz1, axis = 1, keepdims=True)  ## (3,1)
                ## W1: (3,m), dLW1: (3,m), X:(n, m)
                dLW1 = 1/dLz1.shape[1] * dLz1 @ X_c.T
    
                ## ======= parameter update
                vW3 = self.momentum * vW3 + (1-self.momentum) * dLW3
                self.W3_c = self.W3_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vW3
                vb3 = self.momentum * vb3 + (1-self.momentum) * dLb3
                self.b3_c = self.b3_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vb3
                vW2 = self.momentum * vW2 + (1-self.momentum) * dLW2
                self.W2_c = self.W2_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vW2
                vb2 = self.momentum * vb2 + (1-self.momentum) * dLb2
                self.b2_c = self.b2_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vb2
                vW1 = self.momentum * vW1 + (1-self.momentum) * dLW1
                self.W1_c = self.W1_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vW1
                vb1 = self.momentum * vb1 + (1-self.momentum) * dLb1
                self.b1_c = self.b1_c - lr_raito * self.lr_cosine_annealing(i_epoch, deep_iter_limit) * vb1
            
            i_epoch += 1

            z1_e = self.W1_c @ self.X.T + self.b1_c  ## (3,n)
            a1_e = self.relu(z1_e)
    
            z2_e = self.W2_c @ a1_e + self.b2_c   ## (2,n)
            a2_e = self.relu(z2_e)
    
            z3_e = self.W3_c @ a2_e + self.b3_c   ## (1,n)
            a3_e = self.safe_sigmoid(z3_e)

            loss = - np.sum(self.Y.T * np.log(a3_e) + (1-self.Y.T) * np.log(1-a3_e))
            loss_diff = np.abs(loss - loss_prev)
            loss_prev = loss
            if i_epoch % self.output_gap == 1:
                print(f"epoch No.{i_epoch}, loss = {loss:.6f}")
            
            ## terminiate condition
            if i_epoch > deep_iter_limit*2:
                break
            if loss_diff < self.epsilon:
                print("converged")
                break

        return True

    def load_parameters(self, W1, b1, W2, b2, W3, b3):
        self.W1_c = W1
        self.b1_c = b1
        self.W2_c = W2
        self.b2_c = b2
        self.W3_c = W3
        self.b3_c = b3
        return True

    def predict(self, Z):
        Z_c = Z.T
        z1 = self.W1_c @ Z_c + self.b1_c  ## (3,n)
        a1 = self.relu(z1)

        z2 = self.W2_c @ a1 + self.b2_c   ## (2,n)
        a2 = self.relu(z2)

        z3 = self.W3_c @ a2 + self.b3_c   ## (1,n)
        a3 = self.safe_sigmoid(z3)

        res = np.where(a3 > 0.5, 1, 0)
        return res.T

    def check_performance(self, X_t, Y_t):
        X_c = X_t.T
        z1 = self.W1_c @ X_c + self.b1_c  ## (3,n)
        a1 = self.relu(z1)

        z2 = self.W2_c @ a1 + self.b2_c   ## (2,n)
        a2 = self.relu(z2)

        z3 = self.W3_c @ a2 + self.b3_c   ## (1,n)
        a3 = self.safe_sigmoid(z3)

        loss = - np.sum(Y_t * np.log(a3) + (1-Y_t) * np.log(1-a3))
        res = np.where(a3 > 0.5, 1, 0)
        return np.sum(res == Y_t) / Y_t.shape[0]