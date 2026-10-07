"""Testes do esquema do artigo v3 e do esquema corrigido com passo parabólico."""
import json, math, sys, time
import numpy as np
from via_sq.diferencas_finitas import FD, taylor_green, enstrofia_espectral

R = {}

# 1. Balanço exato de energia (Teorema 7.1) no esquema do artigo, passo dt = h
f = FD(16, 1 / 400, "artigo"); u = f.projeta(taylor_green(16)); dt = f.h
res = []
for m in range(5):
    E0, D0 = f.energia(u), f.dissipacao(u)
    u = f.passo(u, dt)
    res.append(abs(f.energia(u) - (E0 - dt * D0)) / E0)
R["balanco_artigo_residuo_relativo"] = res

# 2. Estabilidade do esquema do artigo (dt = h) e com passo parabólico
est = {}
for N in (16, 32, 48):
    for nome, dtf in (("dt=h", lambda h: h), ("dt parabolico", lambda h: min(0.2 * h, 0.15 * h * h * 400))):
        f = FD(N, 1 / 400, "artigo"); u = f.projeta(taylor_green(N)); dt = dtf(f.h)
        n = int(2.0 / dt); ok = True
        for _ in range(n):
            u = f.passo(u, dt)
            if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
                ok = False; break
        est[f"N={N} {nome}"] = {"estavel_ate_t2": ok, "max_u": float(np.abs(u).max()) if np.isfinite(u).all() else None}
R["estabilidade_artigo"] = est

# 3. Conservação de energia do esquema corrigido sem viscosidade
f = FD(32, 0.0, "corrigido"); u = f.projeta(taylor_green(32)); E0 = f.energia(u)
for _ in range(100):
    u = f.passo(u, 0.01)
R["corrigido_euler_erro_energia"] = abs(f.energia(u) - E0) / E0

# 4. Convergência do corrigido para o solucionador espectral (TG, Re=400; pico espectral 128³ = 2,1916)
conv = {}
for N in [int(x) for x in sys.argv[1:]] or [32, 48, 64]:
    f = FD(N, 1 / 400, "corrigido"); u = f.projeta(taylor_green(N))
    dt = min(0.2 * f.h, 0.15 * f.h ** 2 * 400); n = int(round(8 / dt)); dt = 8 / n
    t0 = time.time(); Z = [enstrofia_espectral(u)]; E = [f.energia(u)]
    for m in range(n):
        u = f.passo(u, dt)
        if (m + 1) % max(1, n // 80) == 0:
            Z.append(enstrofia_espectral(u)); E.append(f.energia(u))
    conv[N] = {"Zmax": max(Z), "Z0": Z[0], "Efinal": E[-1], "energia_monotona": all(b <= a * (1 + 1e-12) for a, b in zip(E, E[1:])), "s": time.time() - t0}
    print(N, conv[N], flush=True)
R["convergencia_corrigido"] = conv
R["referencia_espectral_Zmax_128"] = 2.1916
json.dump(R, open("teste_parabola.json", "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "convergencia_corrigido"}, indent=1))
