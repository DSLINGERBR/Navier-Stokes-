"""Roda todos os experimentos e grava resultados.json."""
import json, math, time
import numpy as np
from via_sq.leonardo import (PHI, DELTA, HUMANO, MATERIAIS, Parafuso, Ornitoptero, parafuso_aureo, parafuso_prata,
                  ornitoptero_aureo, ornitoptero_prata, menor_escala)
from via_sq.espectral import Fluxo, CONDICOES


def limpa(x):
    if isinstance(x, dict):
        return {k: limpa(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpa(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def aero():
    out = {"parafuso": {}, "ornitoptero": {}, "familias_parafuso": [], "familias_ornitoptero": [], "escala": {}}
    for f in (parafuso_aureo, parafuso_prata):
        p = f()
        out["parafuso"][p.nome] = {"geometria": {"D_m": p.D, "passo_m": p.passo, "cubo_frac": p.cubo,
                                                 "angulo_ponta_graus": math.degrees(p.angulo_passo(p.R))},
                                    **{m: p.pairado(m) for m in MATERIAIS}}
    for f in (ornitoptero_aureo, ornitoptero_prata):
        o = f()
        out["ornitoptero"][o.nome] = {"geometria": {"envergadura_m": o.envergadura, "alongamento": o.alongamento, "area_m2": o.area},
                                       **{m: o.voo(m) for m in MATERIAIS}}
    # tentativa e erro dentro de cada família: potências k de δ e de φ
    for fam, base in (("prata", DELTA), ("áurea", PHI)):
        for kp in range(0, 4):
            for kc in range(1, 4):
                p = Parafuso(f"{fam} passo D/{fam[0]}^{kp} cubo 1/{fam[0]}^{kc}", 4.8, 4.8 / base ** kp, 1 / base ** kc)
                r = p.pairado("moderno")
                out["familias_parafuso"].append({"familia": fam, "k_passo": kp, "k_cubo": kc,
                                                 "potencia_W": r["potencia_W"], "figura_merito": r["figura_merito"]})
        for k in range(1, 5):
            o = Ornitoptero(f"{fam} AR={fam[0]}^{k}", 10.0, base ** k)
            r = o.voo("moderno")
            out["familias_ornitoptero"].append({"familia": fam, "k": k, "alongamento": base ** k, "potencia_W": r["potencia_W"]})
    # que escala seria necessária para voar com força humana (materiais modernos)?
    for alvo_nome, alvo in HUMANO.items():
        for nome, fab, av, lo, hi in (
            ("parafuso áureo", parafuso_aureo, lambda p: p.pairado("moderno"), 2, 150),
            ("parafuso de prata", parafuso_prata, lambda p: p.pairado("moderno"), 2, 150),
            ("ornitóptero áureo", ornitoptero_aureo, lambda o: o.voo("moderno"), 4, 120),
            ("ornitóptero de prata", ornitoptero_prata, lambda o: o.voo("moderno"), 4, 120)):
            x, pmin = menor_escala(fab, av, alvo, lo, hi)
            out["escala"].setdefault(nome, {})[alvo_nome] = {"dimensao_m": x, "potencia_minima_na_busca_W": pmin}
    # curva potência × dimensão para o artigo
    out["curvas"] = {
        "D_m": list(np.geomspace(2, 150, 30)),
        "parafuso_aureo_W": [parafuso_aureo(D).pairado("moderno")["potencia_W"] for D in np.geomspace(2, 150, 30)],
        "parafuso_prata_W": [parafuso_prata(D).pairado("moderno")["potencia_W"] for D in np.geomspace(2, 150, 30)],
        "b_m": list(np.geomspace(4, 120, 30)),
        "ornitoptero_aureo_W": [ornitoptero_aureo(b).voo("moderno")["potencia_W"] for b in np.geomspace(4, 120, 30)],
        "ornitoptero_prata_W": [ornitoptero_prata(b).voo("moderno")["potencia_W"] for b in np.geomspace(4, 120, 30)],
    }
    return out


def navier_stokes():
    out = {}
    for N in (32, 64):
        f = Fluxo(N=N, nu=1 / 400)
        conds = CONDICOES if N == 32 else CONDICOES[:2]
        for c in conds:
            t0 = time.time()
            U, h = f.roda(f.condicao(c), T=8.0, dt=0.01 if N == 32 else 0.005, cada=20 if N == 32 else 40)
            h["espectro_final"] = list(f.espectro(U))
            h["segundos"] = time.time() - t0
            out[f"{c} | N={N}"] = h
            print(c, N, round(h["segundos"], 1), "Zmax", round(max(h["Z"]), 3), "cauda max", max(h["cauda"]))
    # validação: Euler (nu=0) conserva energia
    f = Fluxo(N=32, nu=0.0)
    U = f.taylor_green(); E0 = f.energia(U)
    U, h = f.roda(U, T=1.0, dt=0.005, cada=200)
    out["validacao_euler"] = {"E0": E0, "E1": f.energia(U), "erro_rel": abs(f.energia(U) - E0) / E0}
    return out


if __name__ == "__main__":
    r = {"aero": aero()}
    print("aero ok")
    r["navier_stokes"] = navier_stokes()
    json.dump(limpa(r), open("resultados.json", "w"), ensure_ascii=False, indent=1)
    print("gravado")
