import matplotlib.pyplot as plt
from BBN import BBN

# all available indices
index = ['neutron', 'proton', 'deutron', 'tritium',
         'helium3', 'helium4', 'lithium7', 'beryllium7']
label = ['n', 'p', 'd', 't', r'He$^3$',
         r'He$^4$', r'Li$^7$', r'Be$^7$']

# initialize() has been executed with default arguments
T,X = BBN(5e-10, index, atol=1e-13)
plt.axis([2, 1e-2, 1e-13, 2])
plt.loglog(T, X.T)
plt.xlabel('T = temperature / MeV')
plt.ylabel('X = mass fraction')
plt.legend(label)
plt.ylim(1e-12,1e0)


plt.show()
