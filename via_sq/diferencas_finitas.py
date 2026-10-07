"""Esquema de diferenças finitas do artigo v3 e sua versão corrigida (parabólica).

Caixa periódica [0, 2π)³, malha N³, h = 2π/N. A pressão é obtida por projeção
discreta exata (FFT com números de onda modificados do próprio operador).

Artigo:    diferenças avançadas, termo não linear em divergência, Laplaciano com
           pesos 1/i², Euler explícito, passo dt = h.
Corrigido: diferenças centradas, termo não linear antissimétrico, Laplaciano
           padrão, passo parabólico dt = c·h², Runge–Kutta 4.
"""
import math
import numpy as np


class FD:
    def __init__(self, N, nu, versao):
        self.N, self.nu, self.v = N, nu, versao
        self.h = 2 * math.pi / N
        k = np.fft.fftfreq(N, 1 / N)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        h = self.h
        if versao == "artigo":                      # derivada avançada
            self.D = [(np.exp(1j * K * h) - 1) / h for K in (KX, KY, KZ)]
            self.w = [1.0, 1 / 4, 1 / 9]
        else:                                       # derivada centrada
            self.D = [1j * np.sin(K * h) / h for K in (KX, KY, KZ)]
            self.w = [1.0, 1.0, 1.0]
        self.L = -sum(wi * (2 - 2 * np.cos(K * h)) / h ** 2 for wi, K in zip(self.w, (KX, KY, KZ)))
        # projeção: remove a parte gradiente com o par gradiente/divergência do esquema
        self.G = self.D                              # gradiente
        if versao == "artigo":                      # divergência recuada (adjunta do gradiente avançado)
            self.Dv = [(1 - np.exp(-1j * K * h)) / h for K in (KX, KY, KZ)]
        else:
            self.Dv = self.D
        self.den = sum(dv * g for dv, g in zip(self.Dv, self.G))
        self.den[np.abs(self.den) < 1e-14] = 1

    def d(self, f, i):
        return np.real(np.fft.ifftn(self.D[i] * np.fft.fftn(f)))

    def projeta(self, U):
        H = [np.fft.fftn(c) for c in U]
        div = sum(dv * c for dv, c in zip(self.Dv, H))
        phi = div / self.den
        return np.array([np.real(np.fft.ifftn(c - g * phi)) for c, g in zip(H, self.G)])

    def lap(self, f):
        return np.real(np.fft.ifftn(self.L * np.fft.fftn(f)))

    def nao_linear(self, u):
        if self.v == "artigo":                      # N = div(u ⊗ u)
            return np.array([sum(self.d(u[j] * u[i], j) for j in range(3)) for i in range(3)])
        adv = np.array([sum(u[j] * self.d(u[i], j) for j in range(3)) for i in range(3)])
        div = np.array([sum(self.d(u[j] * u[i], j) for j in range(3)) for i in range(3)])
        return 0.5 * (adv + div)                    # forma antissimétrica

    def rhs(self, u):
        return self.projeta(-self.nao_linear(u) + self.nu * np.array([self.lap(c) for c in u]))

    def energia(self, u):
        return 0.5 * float(np.mean(np.sum(u ** 2, axis=0)))

    def dissipacao(self, u):
        return float(self.nu * np.mean(sum(self.w[i] * self.d(u[c], i) ** 2 for c in range(3) for i in range(3))))

    def passo(self, u, dt):
        if self.v == "artigo":
            return u + dt * self.rhs(u)
        k1 = self.rhs(u); k2 = self.rhs(u + .5 * dt * k1); k3 = self.rhs(u + .5 * dt * k2); k4 = self.rhs(u + dt * k3)
        return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def taylor_green(N):
    x = np.arange(N) * 2 * math.pi / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.array([np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), 0 * X])


def enstrofia_espectral(u):
    N = u.shape[1]
    k = np.fft.fftfreq(N, 1 / N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    H = [np.fft.fftn(c) for c in u]
    w = [1j * (KY * H[2] - KZ * H[1]), 1j * (KZ * H[0] - KX * H[2]), 1j * (KX * H[1] - KY * H[0])]
    return 0.5 * float(sum(np.mean(np.real(np.fft.ifftn(c)) ** 2) for c in w))
