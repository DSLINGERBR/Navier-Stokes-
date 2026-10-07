"""Estudo de convergência em malha e de suavidade (faixa de analiticidade)."""
import json, sys, time
from via_sq.espectral import Fluxo

CONDS = ["Taylor–Green", "Prata (cascas de Pell 1, 2, 5)", "Áureo (cascas de Fibonacci 1, 2, 3, 5, 8)"]
saida = "ns_avanco.json"
try:
    R = json.load(open(saida))
except FileNotFoundError:
    R = {}
for N in [int(x) for x in sys.argv[1:]] or [64, 96, 128]:
    f = Fluxo(N=N, nu=1 / 400)
    for c in CONDS:
        chave = f"{c} | N={N}"
        if chave in R:
            continue
        t0 = time.time()
        U, h = f.roda(f.condicao(c), T=8.0, dt=0.01, cada=20)
        h["espectro_final"] = [float(x) for x in f.espectro(U)]
        h["segundos"] = time.time() - t0
        R[chave] = h
        json.dump(R, open(saida, "w"), ensure_ascii=False)
        i = max(range(len(h["Z"])), key=h["Z"].__getitem__)
        print(f"{chave}: {h['segundos']:.0f}s  Zmax={h['Z'][i]:.4f} em t={h['t'][i]:.1f}  "
              f"delta_min={min(h['delta']):.4f}  delta*kmax_min={min(h['delta'])*f.kcorte:.2f}", flush=True)
