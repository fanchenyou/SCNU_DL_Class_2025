import numpy as np, matplotlib.pyplot as plt
from scipy.stats import logistic

mu, sig = -0.4, 0.2
x = np.linspace(0.001, 0.999, 500)
lx = np.log(x/(1-x))          # logit
p = 1/(sig*np.sqrt(2*np.pi)) * 1/(x*(1-x)) * np.exp(-0.5*((lx-mu)/sig)**2)

plt.plot(x, p)
plt.title('Logit-Normal(μ=-0.4, σ=0.2)'); plt.xlabel('x'); plt.ylabel('pdf')
plt.show()