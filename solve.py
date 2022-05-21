import time
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint
import math
# g = 0.3
# ω = 1
# β = 0.03901
# Γ = 0.209
# ϕ = 0


# def f(y, t):
#     x = y[0]
#     p = y[1]
#     χ = y[2]
#     Π = y[3]

#     f0 = p
#     f1 = -β**2 * x**3 + (1-3*β**2*χ**2) * x - 2*Γ*p + g/β * math.cos(ω*t)
#     f2 = Π + Γ*((χ-χ**3+χ*Π**2-1/(4*χ))*cos2ϕ - Π*(-1+2*χ**2)
#                 * sin2ϕ + χ - χ**3 - χ*Π**2 + 1/(4*χ))
#     f3 = χ*(1-3*β**2*(x**2 + χ**2)) + 1/(4*χ**3) + Γ*((Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * cos2ϕ
#                                                       - (-1/(4*χ**2)+1/χ -
#                                                          χ+2*χ*Π**2)*sin2ϕ
#                                                       + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2))
#     return np.array([f0, f1, f2, f3])


# def W(y, t):
#     x = y[0]
#     p = y[1]
#     χ = y[2]
#     Π = y[3]
#     return np.diag([
#         sqrtΓ*(2*(χ ** 2 - 1/2)*cosϕ + 2 * χ*Π*sinϕ),
#         sqrtΓ * (2 * (1/(4*χ**2) + Π**2 - 1/2)*sinϕ+2*χ*Π*cosϕ),
#         0,
#         0])


# def Np(ϕ, χ, Π):
#     return 2 * (1/(4*χ**2) + Π**2 - 1/2)*sinϕ+2*χ*Π*cosϕ


# def Nx(ϕ, χ, Π):
#     return 2*(χ ** 2 - 1/2)*cosϕ + 2 * χ*Π*sinϕ


# def Fχ(ϕ, χ, Π):
#     return (χ-χ**3+χ*Π**2-1/(4*χ))*cos2ϕ - Π*(-1+2*χ**2)\
#         * sin2ϕ + χ - χ**3 - χ*Π**2 + 1/(4*χ)


# def FΠ(ϕ, χ, Π):
#     return (Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * cos2ϕ\
#         - (-1/(4*χ**2)+1/χ-χ+2*χ*Π**2)*sin2ϕ\
#         + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2)


# cosϕ = math.cos(ϕ)
# cos2ϕ = math.cos(2*ϕ)
# sinϕ = math.sin(ϕ)
# sin2ϕ = math.cos(2*ϕ)
# sqrtΓ = math.sqrt(Γ)


# start_time = time.time()
# y0 = np.array([0, 0.1, 0.1, 0.001])
# tspan = np.linspace(0, 5., 100)
# t = np.linspace(0, 40*(2*np.pi)/ω, 160000)
# xs = sdeint.itoint(f, W, y0, t)
# x = [β*xs[4000*i, 0] for i in range(40)]
# y = [β*xs[4000*i, 1] for i in range(40)]
# plt.scatter(x, y, color='blue', s=.4)
# plt.xlabel('βx', fontsize=10)
# plt.ylabel('βy', fontsize=10)
# plt.tick_params(labelsize=10)
# plt.title('The Poincare section')
# plt.savefig("output.png")

# print("--- %s seconds ---" % (time.time() - start_time))

import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint
import math
import threading


class oscillator:

    def solve(self, g, ω, β, Γ, ϕ, y0, t):
        cosϕ = math.cos(ϕ)
        cos2ϕ = math.cos(2*ϕ)
        sinϕ = math.sin(ϕ)
        sin2ϕ = math.cos(2*ϕ)
        sqrtΓ = math.sqrt(Γ)

        def f(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]

            f0 = p
            f1 = -β**2 * x**3 + (1-3*β**2*χ**2) * x - 2 * \
                Γ*p + g/β * math.cos(ω*t)
            f2 = Π + Γ*((χ-χ**3+χ*Π**2-1/(4*χ))*cos2ϕ - Π*(-1+2*χ**2)
                        * sin2ϕ + χ - χ**3 - χ*Π**2 + 1/(4*χ))
            f3 = χ*(1-3*β**2*(x**2 + χ**2)) + 1/(4*χ**3) + Γ*((Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * cos2ϕ
                                                              - (-1/(4*χ**2)+1/χ-χ+2*χ*Π**2)*sin2ϕ
                                                              + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2))
            return np.array([f0, f1, f2, f3])

        def W(y, t):
            x = y[0]
            p = y[1]
            χ = y[2]
            Π = y[3]
            return np.diag([
                sqrtΓ*(2*(χ ** 2 - 1/2)*cosϕ + 2 * χ*Π*sinϕ),
                sqrtΓ * (2 * (1/(4*χ**2) + Π**2 - 1/2)*sinϕ+2*χ*Π*cosϕ),
                0,
                0])

        xs = sdeint.itoint(f, W, y0, t)
        x = [β*xs[4000*i, 0] for i in range(40)]
        y = [β*xs[4000*i, 1] for i in range(40)]
        plt.scatter(x, y, color='blue', s=.4)
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title('The Poincare section')
        plt.savefig("{}.png".format(β))

    def __init__(self, g, ω, β, Γ, ϕ, y0, t):
        thread = threading.Thread(target=self.solve)
        return thread.start(g, ω, β, Γ, ϕ, y0, t)


def Np(ϕ, χ, Π):
    return 2 * (1/(4*χ**2) + Π**2 - 1/2)*sinϕ+2*χ*Π*cosϕ


def Nx(ϕ, χ, Π):
    return 2*(χ ** 2 - 1/2)*cosϕ + 2 * χ*Π*sinϕ


def Fχ(ϕ, χ, Π):
    return (χ-χ**3+χ*Π**2-1/(4*χ))*cos2ϕ - Π*(-1+2*χ**2)\
        * sin2ϕ + χ - χ**3 - χ*Π**2 + 1/(4*χ)


def FΠ(ϕ, χ, Π):
    return (Π**3 - Π + 3*Π/(4*χ**2) - Π*χ**2) * cos2ϕ\
        - (-1/(4*χ**2)+1/χ-χ+2*χ*Π**2)*sin2ϕ\
        + (-Π**3 - Π - 3*Π/(4*χ**2)-Π*χ**2)


g = 0.3
ω = 1
β = 0.03901
Γ = 0.209
ϕ = 0
y0 = np.array([0, 0.1, 0.1, 0.001])
t = np.linspace(0, 40*(2*np.pi)/ω, 160000)
start_time = time.time()
oscillator(g, ω, β, Γ, ϕ, y0, t)
β = 0.00001
oscillator(g, ω, β, Γ, ϕ, y0, t)


print("--- %s seconds ---" % (time.time() - start_time))
