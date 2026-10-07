"""Mesmo esquema do teste C, agora em aritmética intervalar (mpmath.iv):
cada número é um intervalo que contém, com garantia, o valor exato."""
import json, time
from fractions import Fraction as Fr
from mpmath import iv
from mpmath.libmp import to_rational
from via_sq.aritmetica_exata import passo_vorticidade


def extremos(x):
    a, b = x._mpi_
    return Fr(*to_rational(a)), Fr(*to_rational(b))


def roda(N=6, passos=60, prec=53):
    iv.prec = prec
    base = [[(i * 3 + j * 5) % 7 - 3 for j in range(N)] for i in range(N)]
    wi = [[iv.mpf(x) for x in l] for l in base]
    wf = [[float(x) for x in l] for l in base]
    wr = [[Fr(x) for x in l] for l in base]
    hist = []
    for s in range(1, passos + 1):
        t0 = time.time(); wi = passo_vorticidade(wi, N, iv.mpf(1)/10, iv.mpf(1)/100, iv.mpf(1), iv.mpf(2), iv.mpf(0)); ti = time.time() - t0
        wf = passo_vorticidade(wf, N, 0.1, 0.01, 1.0, 2.0, 0.0)
        ok_r = s <= 8
        if ok_r:
            wr = passo_vorticidade(wr, N, Fr(1, 10), Fr(1, 100), Fr(1), Fr(2), Fr(0))
        largura = max(float(x.b - x.a) for l in wi for x in l)
        contem_float = all(x.a <= y <= x.b for li, lf in zip(wi, wf) for x, y in zip(li, lf))
        contem_exato = all(extremos(x)[0] <= r <= extremos(x)[1] for li, lr in zip(wi, wr) for x, r in zip(li, lr)) if ok_r else None
        if s in (1, 2, 4, 8, 15, 30, 60):
            hist.append({"passo": s, "largura_max": largura, "tempo_s": ti, "contem_valor_exato": contem_exato, "contem_float": contem_float})
    return hist

if __name__ == "__main__":
    R = {"precisao_53_bits": roda(prec=53), "precisao_106_bits": roda(prec=106)}
    print(json.dumps(R, indent=1))
    json.dump(R, open("teste_intervalar.json", "w"), indent=1)
