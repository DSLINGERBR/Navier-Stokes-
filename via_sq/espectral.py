"""Experimento numérico com Navier–Stokes 3D incompressível.

Método pseudoespectral em caixa periódica [0, 2π)³, regra de 2/3 contra
aliasing, Runge–Kutta 4. Compara condições iniciais de mesma energia:
Taylor–Green, cascas de Pell (prata), cascas de Fibonacci (áureo) e
controle de banda larga. Mede energia, enstrofia, vorticidade máxima e a
fração de energia nas escalas mais finas (indicador de resolução).

Isto é um experimento numérico em resolução finita. Não é prova de
existência ou suavidade global.
"""
import math
import numpy as np
import scipy.fft as sfft

PHI = (1 + 5 ** 0.5) / 2
DELTA = 1 + 2 ** 0.5


class Fluxo:
    def __init__(self, N=32, nu=1 / 400):
        self.N, self.nu = N, nu
        k = np.fft.fftfreq(N, 1 / N)
        kz = np.fft.rfftfreq(N, 1 / N)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, kz, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2
        self.K2i = np.where(self.K2 == 0, 1, self.K2)
        kmax = N // 3
        self.mask = (np.abs(self.KX) < kmax) & (np.abs(self.KY) < kmax) & (np.abs(self.KZ) < kmax)
        self.kmag = np.sqrt(self.K2)
        self.kcorte = kmax
        x = np.arange(N) * 2 * math.pi / N
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing="ij")
        self.peso = np.where(self.KZ == 0, 1.0, 2.0) / N ** 6   # soma de Parseval com rfft

    # --- operadores
    def fft(self, f):
        return sfft.rfftn(f, workers=2)

    def ifft(self, F):
        return sfft.irfftn(F, s=(self.N,) * 3, workers=2)

    def projeta(self, U):
        kx, ky, kz = self.KX, self.KY, self.KZ
        div = (kx * U[0] + ky * U[1] + kz * U[2]) / self.K2i
        return np.array([U[0] - kx * div, U[1] - ky * div, U[2] - kz * div])

    def vort_hat(self, U):
        i = 1j
        return np.array([i * (self.KY * U[2] - self.KZ * U[1]),
                         i * (self.KZ * U[0] - self.KX * U[2]),
                         i * (self.KX * U[1] - self.KY * U[0])])

    def rhs(self, U):
        u = np.array([self.ifft(c) for c in U])
        w = np.array([self.ifft(c) for c in self.vort_hat(U)])
        cr = np.array([u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]])
        NL = np.array([self.fft(c) for c in cr]) * self.mask
        return self.projeta(NL) - self.nu * self.K2 * U

    # --- diagnósticos
    def energia(self, U):
        return 0.5 * float(np.sum(self.peso * np.sum(np.abs(U) ** 2, axis=0)))

    def enstrofia(self, U):
        return 0.5 * float(np.sum(self.peso * np.sum(np.abs(self.vort_hat(U)) ** 2, axis=0)))

    def vort_max(self, U):
        w = np.array([self.ifft(c) for c in self.vort_hat(U)])
        return float(np.sqrt((w ** 2).sum(0)).max())

    def divergencia_max(self, U):
        d = self.ifft(1j * (self.KX * U[0] + self.KY * U[1] + self.KZ * U[2]))
        return float(np.abs(d).max())

    def fracao_cauda(self, U):
        e = self.peso * np.sum(np.abs(U) ** 2, axis=0)
        cauda = e[(self.kmag >= self.kcorte - 2) & self.mask].sum()
        return float(cauda / max(e.sum(), 1e-300))

    def faixa_analiticidade(self, U):
        """Largura delta da faixa de analiticidade: E(k) ~ exp(-2 delta k) na cauda.
        delta -> 0 em tempo finito seria a assinatura de uma singularidade.
        A simulação só é confiável enquanto delta * kmax > 1."""
        E = self.espectro(U)
        k = np.arange(len(E))
        sel = (k >= self.kcorte // 2) & (k <= self.kcorte - 1) & (E > 0)
        inc = np.polyfit(k[sel], np.log(E[sel]), 1)[0]
        return float(-inc / 2)

    def espectro(self, U):
        e = 0.5 * self.peso * np.sum(np.abs(U) ** 2, axis=0)
        kb = np.rint(self.kmag).astype(int)
        return np.bincount(kb.ravel(), e.ravel())[: self.kcorte + 1]

    # --- condições iniciais (energia normalizada para E0)
    def normaliza(self, U, E0=0.125):
        U = np.array([self.fft(self.ifft(c)) for c in U])
        U = self.projeta(U) * self.mask
        return U * math.sqrt(E0 / self.energia(U))

    def taylor_green(self):
        u = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        v = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        return self.normaliza(np.array([self.fft(u), self.fft(v), self.fft(0 * u)]))

    def cascas(self, raios, pesos, semente=7):
        """Mesmo campo inicial em qualquer malha: gerado em 32³ e transportado espectralmente."""
        if self.N != 32:
            ref = Fluxo(N=32, nu=self.nu).cascas(raios, pesos, semente)
            U = np.zeros((3,) + self.K2.shape, dtype=complex)
            idx = [i for i in range(-15, 16)]
            src = [i % 32 for i in idx]
            dst = [i % self.N for i in idx]
            for a, sa in zip(dst, src):
                for b, sb in zip(dst, src):
                    U[:, a, b, :16] = ref[:, sa, sb, :16]
            return self.normaliza(U * (self.N / 32) ** 3)
        rng = np.random.default_rng(semente)
        fase = np.exp(2j * math.pi * rng.random((3,) + self.K2.shape))
        amp = np.zeros(self.K2.shape)
        for r, a in zip(raios, pesos):
            amp += a * (np.abs(self.kmag - r) < 0.5)
        return self.normaliza(fase * amp)

    def condicao(self, nome):
        if nome == "Taylor–Green":
            return self.taylor_green()
        if nome == "Prata (cascas de Pell 1, 2, 5)":
            return self.cascas([1, 2, 5], [DELTA ** -n for n in range(3)])
        if nome == "Áureo (cascas de Fibonacci 1, 2, 3, 5, 8)":
            return self.cascas([1, 2, 3, 5, 8], [PHI ** -n for n in range(5)])
        if nome == "Controle (banda larga 1 a 8)":
            return self.cascas(list(range(1, 9)), [1.0] * 8)
        raise KeyError(nome)

    def roda(self, U, T=8.0, dt=0.01, cada=20):
        hist = {"t": [], "E": [], "Z": [], "wmax": [], "cauda": [], "div": [], "delta": []}
        passos = int(round(T / dt))
        for n in range(passos + 1):
            if n % cada == 0:
                hist["t"].append(n * dt); hist["E"].append(self.energia(U)); hist["Z"].append(self.enstrofia(U))
                hist["wmax"].append(self.vort_max(U)); hist["cauda"].append(self.fracao_cauda(U)); hist["div"].append(self.divergencia_max(U)); hist["delta"].append(self.faixa_analiticidade(U))
            if n == passos:
                break
            k1 = self.rhs(U); k2 = self.rhs(U + 0.5 * dt * k1); k3 = self.rhs(U + 0.5 * dt * k2); k4 = self.rhs(U + dt * k3)
            U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return U, hist


CONDICOES = ["Taylor–Green", "Prata (cascas de Pell 1, 2, 5)", "Áureo (cascas de Fibonacci 1, 2, 3, 5, 8)", "Controle (banda larga 1 a 8)"]
