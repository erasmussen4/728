class Abundance():
    def __init__(self, T0):
        #Constants
        self.T0 = T0
        self.elem = {0: "n", 1: "p", 2: "2H", 3: "3H", 4: "3He", 5: "4He", 6: "7Li", 7: "7Be"}
        self.A = {"n": 1, "p": 1, "2H": 2, "3H": 3, "3He": 3, "4He": 4, "7Li": 7, "7Be": 7}
        self.B = {"n": 0, "p": 0, "2H": 2.22, "3H": 8.48, "3He": 7.72, "4He": 28.3, "7Li": 39.25, "7Be": 37.6}
        self.gA = {"n": 2, "p": 2, "2H": 3, "3H": 2, "3He": 2, "4He": 1, "7Li": 4, "7Be": 4}
        self.h = 1.15e-5
        self.H0 = 1
        self.me = 0.510998911
        self.G = 6.70881e-45
        self.hbar = 6.58211899e-22

        #Lists and Stuff 
        self.A = {Ap, An, AD, AT, AHe3, AHe4, ALi7, ABe7}
        self.X_A = {Xp/Ap, Xn/An, XD/AD, XT/AT, XHe3/AHe3, XHe4/AHe4, XLi7/ALi7, XBe7/ABe7}
        self.node,self.weights = roots_laguerre(64)
        self.int_rates_a = [
            [2.5e4, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [2.23e3, 0, 1, 0.112, 3.38, 2.65, 0, 0, 0, -3.72],
            [1, 0, 0, 0, 1, 75.5, 0, 0, 1250, 0, 0],
            [7.06e8, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [2.87e4, 0, 1, 0.108, 0.466, 0.352, 0.3, 0.576, 0, -3.87],
            [6e3, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [3.9e8, 0, 1, 0.0979, 0.642, 0.44, 0, 0, 0, -4.26],
            [3.9e8, 0, 1, 0.0979, 0.642, 0.44, 0, 0, 0, -4.26],
            [24.1, 0, 0, 0, 1, .685, .152, .265, 0, -4.26],
            [2.6e9, 1, 0, 0, 0, 0, 0, 0, -2.99, 0],
            [1.38e9, 1, 0, 0, 0, 0, 0, 0, -0.745, 0],
            [1.19e10, 0, 1, 0.034, 0, 0, 0, 0, 0, -12.25],
            [1.10e9, 0, 1, 0.0857, 0, 0, 0, 0, 0, -4.87],
            [5.60e9, 0, 1, 0.054, 0, 0, 0, 0, 0, -7.72],
            [3.88e9, 0, 1, 0.054, 0, 0, 0, 0, 0, -7.72],
            [4.8e6, 0, 1, 0.0326, -.219, -.0499, .0258, .015, 0, -12.8],
            [5.28e5, 0, 1, 0.0516, 0, 0, 0, 0, 0, -8.08],
            [6.74e9, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1.42e9, 0, 1, 0.0493, 0, 0, 0, 0, 0, -8.47],
            [1.2e7, 0, 0, 0, 0, 0, 0, 1, 0, 0] 
        ]
        self.int_rates_b = [ 
            [4.68e9, -1, 3/2, -25.82],
            [1.63e10, -1, 3/2, -63.75], 
            [1.63e10, -1, 3/2, -72.62], 
            [1, 0, 0, -8.864], 
            [2.59e10, -1, 3/2, -229.9], 
            [2.6e10, -1, 3/2, -238.8], 
            [1.73, 0, 0, -37.94], 
            [1.73, 0, 0, -46.8], 
            [4.5e10, -1, 3/2, -276.7], 
            [5.5, 0, 0, -213], 
            [5.5, 0, 0, -204.1], 
            [3.37e-10, 1, -3/2, -149.2], 
            [3.37e-10, 1, -3/2, -131.5], 
            [3.37e-10, 1, -3/2, -140.4], 
            [1.59, 0, 0, -166.2], 
            [1.12e10, -1, 3/2, -18.42], 
            [1.12e10, -1, 3/2, -28.63], 
            [1, 0, 0, -19.07], 
            [4.64, 0, 0, -201.3], 
            [4.64, 0, 0, -220.4]
        ]
    def electron_gas(self, T): 
        """
            Calculates rho_e, P_e, and c_e for any given temperature.
        """
        a = np.expand_dims(self.me/T, -1)
        x = self.node
        x2 = x*x
        y = np.sqrt(x*x + a*a)
        z = np.exp(y)
        f = np.asarray([y, x2/y/3, y*y*z/(z+1)])
        f*= x2/(z+1)*np.exp(x)
        f = np.dot(f, self.weight)
        f*= T**4*2/np.pi**2 
        return f 
    
    def rho(self, T):
        """
            Calculates rho for any given temperature.
        """
        N_nu = 3
        rho_r = np.pi**2/15*T**4
        rho_nu = 7/8*np.pi**2/15*N_nu*T**4
        rho_e = self.electron_gas(T)[0]
        rho_total = rho_e + rho_nu + rho_r
        return rho_total
    
    def dr_dT(self,T):
        """
            Calculates dr/dT for any given temperature.
        """
        rho_r = np.pi**2/15*T**4
        P_e = self.electron_gas(T)[1] 
        P_r = rho_r/15*T**4/3
        P = P_e + P_r 
        c_e = self.electron_gas(T)[2]
        c_r = rho_r*4
        c = (c_e + c_r)/T
        return -c/(P + self.rho(T))
    
    def H(self, T):
        """
            Calculates the value of Hubble at any given time.
        """
        return np.sqrt(8*np.pi*self.G/3)*np.sqrt(self.rho(T))/self.hbar
    
    def dt_dT(sefl, T):
        """
            Calculates dt/dT for any given temperature.
        """ 
        return self.dr_dT(T)/(3*self.H)
    
    def X_eq(self, elemKey, T):
        """
            Calculates dr/dT for any given temperature.
        """
        A = self.A[elemKey]
        zeta = zeta(3)
        g = self.gA[elemKey]
        prefactor = g * zeta**(A-1) * 2**((3*A-5)/2) * np.pi**((1-A)/2) * A**(5/2)
        return 
    
    def rho_b(self, T):
        """
            Calculates the baryon density.

            Parameters:
                T: (float), temperature
            Returns:
                rho_b: (float), baryon density
        """
        return self.h * T**3
    
    def interaction_rates_a (self, reactionIdx, T):
        """
            Gives the interaction rate equation for reaction.

            Parameters:
                reactionIdx: (int), index for reaction equation
            Returns:
                int_rate: (float), interaction rate for given T, rho_b 
        """
        rho_b = self.rho_b(T)
        a = self.int_rates_a[reactionIdx,0]
        b = self.int_rates_a[reactionIdx,1:8]
        c = self.int_rates_a[reactionIdx,8:]
        polynomial = b[0]*T**(-3/2) + sum([b[i+1]*T**((i-2)/3) for i in range(len(b)-1)])
        exponential = np.exp(c[0]*T**(-1) + c [1]*(T**(-1/3)))
        return a * rho_b * polynomial * exponential
    
    def interaction_rates_b (self, reactionIdx, T):
        """
            Gives the f_i equation for reaction. 
            
            Parameters: 
                fIdx: (int), index for the f_i equations
            Returns: 
                f_i: (float), f_F for the given T, rho_b
        """ 
        rho_b = self.rho_b(T)
        a,b,c,d = self.int_rates_b[reactionIdx]

        return a * rho_b**(b) * T**c * np.exp(d*T**(-1)) * self.interaction_rates_a(reactionIdx, T, rho_b)
    
    def dXA_dt(self, T):
        """
            Calculate the derivatives from the rate equations.

        """
        dX0_dt = self.X_A[2]*self.interaction_rates_b(0,T) + self.X_A[4]*self.interaction_rates_b(1,T) + self.X_A[1]*self.X_A[4]*self.interaction_rates_a(3,T) + self.X_A[5]*self.interaction_rates_b(4,T) + .5*self.X_A[2]*self.interaction_rates_a(7,T) + self.X_A[2]*self.X_A[4]*self.interaction_rates_a(9,T) + .5*self.X_A[4]**2*self.interaction_rates_a(11,T) + self.X_A[4]*self.X_A[3]*self.interaction_rates_a(13,T) + self.X_A[1]*self.X_A[6]*self.interaction_rates_a(17,T) + .5*self.X_A[5]**2*self.interaction_rates_b(18,T) - self.X_A[0]*(self.X_A[1]*self.interaction_rates_a(0,T) + self.X_A[2]*self.interaction_rates_a(1,T) + self.X_A[3]*self.interaction_rates_b(3,T) + self.X_A[3]*self.interaction_rates_a(4,T) + self.X_A[3]*self.interaction_rates_b(7,T) + self.X_A[5]*self.interaction_rates_b(9,T) + .5*self.X_A[5]*self.X_A[0]*self.interaction_rates_b(11,T) + self.X_A[5]*self.X_A[1]*self.interaction_rates_b(13,T) + self.X_A[6]*self.interaction_rates_b(17,T) + self.X_A[6]*self.interaction_rates_a(18,T))
        dX1_dt = self.X_A[2]*self.interaction_rates_b(0,T) + self.X_A[3]*self.interaction_rates_b(2,T) + self.X_A[3]*self.X_A[0]*self.interaction_rates_b(3,T) + self.X_A[5]*self.interaction_rates_b(5,T) +.5*self.X_A[2]**2*self.interaction_rates_a(6,T) + self.X_A[2]*self.X_A[3]*self.interaction_rates_a(10,T) + self.X_A[3]**2*self.interaction_rates_a(12,T) + self.X_A[3]*self.X_A[4]*self.interaction_rates_a(13,T) + self.X_A[6]*self.X_A[0]*self.interaction_rates_b(17,T) + .5*self.X_A[5]**2*self.interaction_rates_b(19,T) - self.X_A[1]*(self.X_A[0]*self.interaction_rates_a(0,T) + self.X_A[2]*self.interaction_rates_a(2,T) + self.X_A[4]*self.interaction_rates_a(3,T) + self.X_A[4]*self.interaction_rates_a(5,T) + self.X_A[4]*self.interaction_rates_b(6,T) + self.X_A[5]*self.interaction_rates_b(10,T) + self.X_A[1]*self.X_A[5]*self.interaction_rates_b(12,T) + self.X_A[0]*self.X_A[5]*self.interaction_rates_b(13,T) + self.X_A[7]*self.interaction_rates_a(17,T) + self.X_A[7]*self.interaction_rates_a(19,T))
        dX2_dt = self.X_A[0]*self.X_A[1]*self.interaction_rates_a(0,T) + self.X_A[4]*self.interaction_rates_b(1,T) + self.X_A[3]*self.interaction_rates_b(2,T) + 2*self.X_A[4]*self.interaction_rates_b(6,T) + 2*self.X_A[3]*self.interaction_rates_b(7,T) + self.X_A[5]*(2*self.interaction_rates_b(8,T) +self.interaction_rates_b(9,T) + self.interaction_rates_b(10,T)) + self.X_A[4]*self.X_A[3]*self.interaction_rates_a(14,T) - self.X_A[2]*(self.interaction_rates_b(0,T) + self.X_A[0]*self.interaction_rates_a(1,T) + self.X_A[1]*self.interaction_rates_a(2,T) + self.X_A[2]*(self.interaction_rates_a(6,T) + self.interaction_rates_a(7,T) + self.interaction_rates_a(8,T)) + self.X_A[4]*self.interaction_rates_a(9,T) + self.X_A[3]*self.interaction_rates_a(10,T) + self.X_A[5]*self.interaction_rates_b(14,T)) 
        dX3_dt = self.X_A[1]*self.X_A[2]*self.interaction_rates_a(2,T) + self.X_A[1]*self.X_A[4]*self.interaction_rates_a(3,T) + self.X_A[5]*self.interaction_rates_b(4,T) + .5*self.X_A[2]**2*self.interaction_rates_a(7,T) + self.X_A[5]*self.X_A[1]*(self.interaction_rates_b(10,T) + self.X_A[1]*self.interaction_rates_b(12,T) + self.X_A[0]*self.interaction_rates_b(13,T)) + self.X_A[5]*self.X_A[2]*self.interaction_rates_b(14,T) + self.X_A[6]*self.interaction_rates_b(16,T) - self.X_A[3]*(self.interaction_rates_b(2,T) + self.X_A[0]*(self.interaction_rates_b(3,T) + self.interaction_rates_a(4,T) + self.interaction_rates_b(7,T)) + self.X_A[2]*self.interaction_rates_a(10,T) + self.X_A[3]*self.interaction_rates_a(12,T) + self.X_A[4]*(self.interaction_rates_a(13,T) + self.interaction_rates_a(14,T)) + self.X_A[5]*self.interaction_rates_a(16,T))
        dX4_dt = self.X_A[0]*self.X_A[2]*self.interaction_rates_a(1,T) + self.X_A[0]*self.X_A[3]*self.interaction_rates_b(3,T) + self.X_A[5]*self.interaction_rates_b(5,T) + .5*self.X_A[2]**2*self.interaction_rates_a(6,T) + self.X_A[5]*self.X_A[0]*self.interaction_rates_b(9,T) + self.X_A[5]*self.X_A[0]**2*self.interaction_rates_b(11,T) + self.X_A[5]*self.X_A[0]*self.X_A[1]*self.interaction_rates_b(13,T) + self.X_A[5]*self.X_A[2]*self.interaction_rates_b(14,T) * self.X_A[7]*self.interaction_rates_b(15,T) - self.X_A[4]*(self.interaction_rates_b(1,T) + self.X_A[1]*(self.interaction_rates_a(3,T) + self.interaction_rates_a(5,T) + self.interaction_rates_b(6,T)) + self.X_A[2]*self.interaction_rates_a(9,T) + self.X_A[4]*self.interaction_rates_a(11,T) + self.X_A[3]*(self.interaction_rates_a(13,T) + self.interaction_rates_a(14,T)) + self.X_A[5]*self.interaction_rates_a(15,T))
        dX5_dt = self.X_A[0]*self.X_A[3]*self.interaction_rates_a(4,T) + self.X_A[1]*self.X_A[4]*self.interaction_rates_a(5,T) + .5*self.X_A[2]**2*self.interaction_rates_a(8,T) + self.X_A[2]*self.X_A[4]*self.interaction_rates_a(9,T) + self.X_A[2]*self.X_A[3]*self.interaction_rates_a(10,T) + .5*self.X_A[4]**2*self.interaction_rates_a(11,T) + .5*self.X_A[3]**2*self.interaction_rates_a(12,T) + self.X_A[4]*self.X_A[3]*(self.interaction_rates_a(13,T) + self.interaction_rates_a(14,T)) + self.X_A[7]*self.interaction_rates_b(15,T) + self.X_A[6]*self.interaction_rates_b(16,T) + 2*self.X_A[0]*self.X_A[6]*self.interaction_rates_a(18,T) + 2*self.X_A[1]*self.X_A[7]*self.interaction_rates_a(19,T) - self.X_A[5]*(self.interaction_rates_b(4,T) + self.interaction_rates_b(5,T) + self.interaction_rates_b(8,T) + self.X_A[0]*self.interaction_rates_b(9,T) + self.X_A[1]*self.interaction_rates_b(10,T) + self.X_A[0]**2*self.interaction_rates_b(11,T) + self.X_A[1]**2*self.interaction_rates_b(12,T) + self.X_A[0]*self.X_A[1]*self.interaction_rates_b(13,T) + self.X_A[2]*self.interaction_rates_b(14,T) + self.X_A[4]*self.interaction_rates_a(15,T) + self.X_A[3]*self.interaction_rates_a(16,T) + self.X_A[5]*(self.interaction_rates_b(18,T) + self.interaction_rates_b(19,T)))
        dX6_dt = self.X_A[3]*self.X_A[5]*self.interaction_rates_a(16,T) + self.X_A[1]*self.X_A[7]*self.interaction_rates_a(17,T) + .5*self.X_A[5]**2*self.interaction_rates_b(18,T) - self.X_A[6]*(self.interaction_rates_b(16,T) + self.X_A[0]*self.interaction_rates_b(17,T) + self.X_A[0]*self.interaction_rates_a(18,T))
        dX7_dt = self.X_A[4]*self.X_A[5]*self.interaction_rates_a(15,T) + self.X_A[0]*self.X_A[6]*self.interaction_rates_b(17,T) + .5*self.X_A[5]**2*self.interaction_rates_b(19,T) - self.X_A[7]*(self.interaction_rates_b(15,T) + self.X_A[1]*(self.interaction_rates_a(17,T) + self.interaction_rates_a(18,T))) 

        return [dX0_dt, dX1_dt, dX2_dt, dX3_dt, dX4_dt, dX5_dt, dX6_dt, dX7_dt]

    