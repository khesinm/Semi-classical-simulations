import matplotlib.pyplot as plt
import numpy as np
import sdeint

g = 0.1
ω = .5
β = 0.5
Γ = 0.1

y0 = np.array([0, 0.1, 0.1, 0.001])
tspan = np.linspace(0, 5., 1000)


def f(y, t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]
    
    f0 = p
    f1 = -β**2 * x**3 + (1-3*β**2*χ**2) *x - 2*Γ*p + g/β * np.cos(ω*t)
    f2 = Π + Γ*( χ - χ**3 - χ * Π**2 + 1/(4*χ))
    f3 = χ*(-3*β**2 *x**2 + 1) + 1/(4*χ**3) -Γ*Π*(1 + Π**2 + χ**2 + 3/(4*χ**2))
    return np.array([f0, f1, f2, f3])


def GG(y, t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]
    return np.diag([
        2* np.sqrt(Γ)*χ**2 -1/2,
        2*np.sqrt(Γ) * χ* Π,
        0,
        0])
def GG2 (y,t):
    x = y[0]
    p = y[1]
    χ = y[2]
    Π = y[3]
    return np.diag([
        - 2* np.sqrt (Γ) * χ * Π,
        - 2* np.sqrt (Γ) * (1/2 - Π**2 - 1/(4*χ**2)),
        0,
        0,
    ])

def ξ (y,t):
    return GG(y,t) + GG2(y,t)



result = sdeint.itoint(f, ξ, y0, tspan)

result
plt.plot(result)
plt.legend(["x","p", "χ" , "Π"])
fullArray= [[],[],[],[]]
for i in range (len(result)):
    fullArray[0].append(result[i][0])
    fullArray[1].append(result[i][1])
    fullArray[2].append(result[i][2])
    fullArray[3].append(result[i][3])            
plt.show()
