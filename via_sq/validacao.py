"""Validação do solucionador e certificado de erro por simulação.

1. Soluções exatas de Navier–Stokes 3D (escoamentos de Beltrami: ABC e
   ABC em k = 2). Para elas u(t) = u(0) * exp(-nu k^2 t) exatamente, então o
   erro do solucionador é medido diretamente.
2. Taylor–Green com Re = 1600 até t = 10: caso de referência internacional
   (pico de dissipação ~0,0127 perto de t ~ 9 na literatura, com 512³).
3. Certificado: a partir da faixa de analiticidade delta, o erro de
   truncamento espectral estimado é exp(-delta * kmax). Cada execução declara
   se está resolvida (delta * kmax >= 2) e a estimativa de erro.
"""
import json, math, sys, time
import numpy as np
from via_sq.espectral import Fluxo


def certificado(f, hist):
    d = [x for t, x in zip(hist["t"], hist["delta"]) if t >= 0.5] or hist["delta"]
    dk = min(d) * f.kcorte
    return {"delta_min": min(d), "delta_kmax_min": dk, "erro_truncamento_estimado": math.exp(-dk),
            "resolvida": dk >= 2}


def beltrami(f, k=1, A=1.0, B=1.0, Cc=1.0):
    X, Y, Z = f.X, f.Y, f.Z
    u = A * np.sin(k * Z) + Cc * np.cos(k * Y)
    v = B * np.sin(k * X) + A * np.cos(k * Z)
    w = Cc * np.sin(k * Y) + B * np.cos(k * X)
    return np.array([f.fft(u), f.fft(v), f.fft(w)])


def teste_exato(N=32, nu=0.01, T=2.0, dt=0.01, k=1):
    f = Fluxo(N=N, nu=nu)
    U0 = beltrami(f, k)
    U, h = f.roda(U0, T=T, dt=dt, cada=10 ** 9)
    exata = U0 * math.exp(-nu * k * k * T)
    num = np.array([f.ifft(c) for c in U]); ex = np.array([f.ifft(c) for c in exata])
    return {"N": N, "k": k, "nu": nu, "T": T, "erro_max": float(np.abs(num - ex).max()),
            "erro_rel": float(np.abs(num - ex).max() / np.abs(ex).max())}


def taylor_green_1600(N, T=10.0, dt=0.01):
    f = Fluxo(N=N, nu=1 / 1600)
    U0 = f.taylor_green()          # energia 0,125, u0 = 1: Re = 1/nu = 1600
    t0 = time.time()
    U, h = f.roda(U0, T=T, dt=dt, cada=20)
    h["dissipacao"] = [2 * f.nu * z for z in h["Z"]]
    h["segundos"] = time.time() - t0
    h["certificado"] = certificado(f, h)
    return h


if __name__ == "__main__":
    R = {}
    try:
        R = json.load(open("validacao.json"))
    except FileNotFoundError:
        pass
    if "exatas" not in R:
        R["exatas"] = [teste_exato(32, 0.01, 2.0, 0.01, 1), teste_exato(32, 0.01, 2.0, 0.01, 2),
                       teste_exato(32, 0.01, 2.0, 0.005, 2)]
        json.dump(R, open("validacao.json", "w"), ensure_ascii=False)
        print("exatas", R["exatas"], flush=True)
    for N in [int(x) for x in sys.argv[1:]]:
        chave = f"TG1600 | N={N}"
        if chave in R:
            continue
        h = taylor_green_1600(N)
        R[chave] = h
        json.dump(R, open("validacao.json", "w"), ensure_ascii=False)
        i = int(np.argmax(h["dissipacao"]))
        print(f"{chave}: {h['segundos']:.0f}s  pico eps={h['dissipacao'][i]:.5f} em t={h['t'][i]:.1f}  cert={h['certificado']}", flush=True)
