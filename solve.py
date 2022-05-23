import time
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.integrate import odeint
import math
import concurrent.futures
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class oscillator:

    def solve(self, g, ω, β, Γ, ϕ, y0, t, periods, acc, folder):
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
        start_time = time.time()
        T = 2*np.pi/ω
        xs = sdeint.itoint(f, W, y0, t)
        print("--- %s seconds ---" % (time.time() - start_time))

        x = [β*xs[acc*i, 0] for i in range(periods)]
        y = [β*xs[acc*i, 1] for i in range(periods)]
        plt.scatter(x, y, color='blue', s=periods/(acc*10))
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title('The Poincare section')
        plt.savefig(os.path.join(
            folder, "acc={} gamma={}, beta={}.png".format(acc, Γ, β)))


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


if __name__ == '__main__':
    g = 0.3
    ω = 1
    β = 0.03901
    Γ = 0.110
    ϕ = 0.5
    periods = 400
    acc = 100

    y0 = np.array([0, 0.1, 0.1, 0.001])
    T = 2*np.pi/ω

    t = np.linspace(0, periods*T, acc*periods)

    osc = oscillator()
    osc2 = oscillator()
    # osc.solve(g, ω, β, Γ, ϕ, y0, t, periods, acc)

    with concurrent.futures.ProcessPoolExecutor() as executor:

        cwd = os.getcwd()
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        os.mkdir(os.path.join(cwd, current_time))
        folder = os.path.join(cwd, current_time)
        executor.submit(osc.solve, g, ω, 0.06001,
                        0.110, ϕ, y0, t, periods, acc, folder)
        executor.submit(osc2.solve, g, ω, .00001,
                        0.110, ϕ, y0, t, periods, acc, folder)
        executor.submit(osc2.solve, g, ω, .03901, .209,
                        ϕ, y0, t, periods, acc, folder)
        executor.submit(osc2.solve, g, ω, .00001, .209,
                        ϕ, y0, t, periods, acc, folder)
        executor.submit(osc2.solve, g, ω, .25, .05, ϕ,
                        y0, t, periods, acc, folder)
        executor.submit(osc2.solve, g, ω, .12, .05, ϕ,
                        y0, t, periods, acc, folder)
