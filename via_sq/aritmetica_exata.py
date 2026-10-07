"""Teste empírico da via racional ("ruído zero") do documento.

A. Esquema literal: dt = h = 1/n². Estável quando n cresce?
B. Laplaciano com pesos 1/k²: aproxima o Laplaciano verdadeiro?
C. Aritmética racional exata contra ponto flutuante no mesmo esquema
   (vorticidade 2D periódica): diferença, custo e tamanho dos números.
D. Erro de discretização contra erro de arredondamento, numa solução exata.
"""
import json, math, time
from fractions import Fraction as Fr
import numpy as np


# ---------------- A. estabilidade do passo do documento
def teste_A(nu=1 / 400, passos=400):
    out = []
    for n in range(4, 25, 2):
        h = 1 / n ** 2
        dt = 1 / n ** 2
        M = int(round(1 / h))
        if M > 600:
            M = 600
        x = np.arange(M) * h
        u = np.sin(2 * math.pi * x)[:, None] * np.sin(2 * math.pi * x)[None, :]
        u += 1e-12 * np.random.default_rng(0).standard_normal(u.shape)
        r = nu * dt / h ** 2
        for _ in range(passos):
            lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
            u = u + r * lap
            if not np.isfinite(u).all() or np.abs(u).max() > 1e6:
                break
        out.append({"n": n, "nu_dt_sobre_h2": r, "limite_estavel": 0.25, "max_final": float(np.abs(u).max()),
                    "estavel": bool(np.isfinite(u).all() and np.abs(u).max() <= 1.0 + 1e-9)})
    return out


# ---------------- B. consistência do Laplaciano ponderado
def teste_B():
    out = []
    for N in (16, 32, 64):
        h = 2 * math.pi / N
        x = np.arange(N) * h
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        u = np.sin(X) + np.sin(Y) + np.sin(Z)
        lap_exato = -u
        lap_pesos = 0
        lap_padrao = 0
        for k, ax in enumerate((0, 1, 2), start=1):
            d2 = (np.roll(u, -1, ax) - 2 * u + np.roll(u, 1, ax)) / h ** 2
            lap_pesos = lap_pesos + d2 / k ** 2
            lap_padrao = lap_padrao + d2
        out.append({"N": N, "erro_laplaciano_padrao": float(np.abs(lap_padrao - lap_exato).max()),
                    "erro_laplaciano_pesos_1_k2": float(np.abs(lap_pesos - lap_exato).max())})
    return out


# ---------------- C. racional exato x ponto flutuante
def passo_vorticidade(w, N, nu, dt, um, dois, zero):
    """Vorticidade 2D periódica, diferenças centradas, Euler explícito.
    A corrente psi vem de uma iteração de Jacobi com número fixo de passos,
    para que o esquema use só as quatro operações (exato em racionais)."""
    psi = [[zero] * N for _ in range(N)]
    for _ in range(20):
        psi = [[(psi[(i + 1) % N][j] + psi[i - 1][j] + psi[i][(j + 1) % N] + psi[i][j - 1] + w[i][j]) / 4
                for j in range(N)] for i in range(N)]
    novo = []
    for i in range(N):
        linha = []
        for j in range(N):
            u = (psi[i][(j + 1) % N] - psi[i][j - 1]) / dois
            v = -(psi[(i + 1) % N][j] - psi[i - 1][j]) / dois
            wx = (w[(i + 1) % N][j] - w[i - 1][j]) / dois
            wy = (w[i][(j + 1) % N] - w[i][j - 1]) / dois
            lap = w[(i + 1) % N][j] + w[i - 1][j] + w[i][(j + 1) % N] + w[i][j - 1] - 4 * w[i][j]
            linha.append(w[i][j] + dt * (-(u * wx + v * wy) + nu * lap))
        novo.append(linha)
    return novo


def teste_C(N=6, passos=6):
    base = [[(i * 3 + j * 5) % 7 - 3 for j in range(N)] for i in range(N)]
    wr = [[Fr(x) for x in l] for l in base]
    wf = [[float(x) for x in l] for l in base]
    nu_r, dt_r = Fr(1, 10), Fr(1, 100)
    hist = []
    for s in range(1, passos + 1):
        t0 = time.time(); wr = passo_vorticidade(wr, N, nu_r, dt_r, Fr(1), Fr(2), Fr(0)); tr = time.time() - t0
        t0 = time.time(); wf = passo_vorticidade(wf, N, 0.1, 0.01, 1.0, 2.0, 0.0); tf = time.time() - t0
        dig = max(len(str(x.denominator)) for l in wr for x in l)
        dif = max(abs(float(a) - b) for la, lb in zip(wr, wf) for a, b in zip(la, lb))
        hist.append({"passo": s, "digitos_denominador": dig, "dif_racional_vs_float": dif,
                     "tempo_racional_s": tr, "tempo_float_s": tf})
        if dig > 200000:
            break
    return hist


# ---------------- D. discretização x arredondamento
def teste_D(nu=0.05, T=1.0):
    out = []
    for N in (16, 32, 64):
        h = 2 * math.pi / N
        x = np.arange(N) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        w = 2 * np.sin(X) * np.sin(Y)
        dt = 0.2 * h ** 2 / nu
        n = int(math.ceil(T / dt)); dt = T / n
        for _ in range(n):
            lap = (np.roll(w, 1, 0) + np.roll(w, -1, 0) + np.roll(w, 1, 1) + np.roll(w, -1, 1) - 4 * w) / h ** 2
            w = w + dt * nu * lap
        exata = 2 * np.sin(X) * np.sin(Y) * math.exp(-2 * nu * T)
        w32 = w.astype(np.float32).astype(np.float64)
        out.append({"N": N, "erro_discretizacao": float(np.abs(w - exata).max()),
                    "erro_arredondamento_float64": float(np.finfo(np.float64).eps * n),
                    "efeito_trocar_por_float32": float(np.abs(w - w32).max())})
    return out


if __name__ == "__main__":
    R = {"A": teste_A(), "B": teste_B(), "D": teste_D()}
    print("A", json.dumps(R["A"]), flush=True)
    print("B", json.dumps(R["B"]), flush=True)
    print("D", json.dumps(R["D"]), flush=True)
    R["C"] = teste_C()
    print("C", json.dumps(R["C"]), flush=True)
    json.dump(R, open("teste_racional.json", "w"), ensure_ascii=False, indent=1)
