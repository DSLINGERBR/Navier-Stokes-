"""Verificação rápida dos modelos numéricos (malhas pequenas; cada teste leva segundos)."""
import math
import numpy as np
import pytest
from via_sq.leonardo import parafuso_aureo, parafuso_prata, ornitoptero_aureo, ornitoptero_prata
from via_sq.espectral import Fluxo
from via_sq.validacao import beltrami
from via_sq.burgers import roda as burgers
from via_sq.diferencas_finitas import FD, taylor_green as tg_fd
from via_sq.aritmetica_exata import passo_vorticidade
from fractions import Fraction as Fr


# ---- Parte IV.1: Leonardo (valores do artigo, materiais modernos)
@pytest.mark.parametrize("fab,esperado", [(parafuso_aureo, 5156), (parafuso_prata, 7509)])
def test_parafusos(fab, esperado):
    assert round(fab().pairado("moderno")["potencia_W"]) == esperado


@pytest.mark.parametrize("fab,esperado", [(ornitoptero_aureo, 1094), (ornitoptero_prata, 873)])
def test_ornitopteros(fab, esperado):
    assert round(fab().voo("moderno")["potencia_W"]) == esperado


def test_figura_de_merito_respeita_o_momento():
    for fab in (parafuso_aureo, parafuso_prata):
        assert 0 < fab().pairado("moderno")["figura_merito"] <= 1


# ---- Parte IV.2: solucionador espectral
def test_solucao_exata_de_beltrami():
    f = Fluxo(N=16, nu=0.01)
    U0 = beltrami(f, 1)
    U, _ = f.roda(U0, T=1.0, dt=0.01, cada=10**9)
    num = np.array([f.ifft(c) for c in U]); ex = np.array([f.ifft(c) for c in U0]) * math.exp(-0.01)
    assert np.abs(num - ex).max() / np.abs(ex).max() < 1e-12


def test_euler_conserva_energia():
    f = Fluxo(N=16, nu=0.0)
    U = f.taylor_green(); E0 = f.energia(U)
    U, _ = f.roda(U, T=0.5, dt=0.005, cada=10**9)
    assert abs(f.energia(U) - E0) / E0 < 1e-10


def test_incompressibilidade_e_decaimento():
    f = Fluxo(N=16, nu=0.01)
    U, h = f.roda(f.taylor_green(), T=1.0, dt=0.01, cada=10)
    assert max(h["div"]) < 1e-10
    assert all(b <= a * (1 + 1e-12) for a, b in zip(h["E"], h["E"][1:]))


# ---- Parte V.3: Burgers
def test_burgers_choque_viscoso():
    nu = 0.05
    h = burgers(nu, 512)
    d = min(x for x in h["delta"] if np.isfinite(x))
    assert d / nu >= math.pi
    assert max(h["grad_max"]) < 1 / (2 * nu)


# ---- Parte IV.5: diferenças finitas com passo parabólico
def test_diferencas_finitas_conservam_energia():
    f = FD(16, 0.0, "corrigido"); u = f.projeta(tg_fd(16)); E0 = f.energia(u)
    for _ in range(20):
        u = f.passo(u, 0.01)
    assert abs(f.energia(u) - E0) / E0 < 1e-12


# ---- Parte IV.4: aritmética exata cresce; ponto flutuante coincide
def test_racional_exato_coincide_e_cresce():
    N = 4
    base = [[(i * 3 + j * 5) % 7 - 3 for j in range(N)] for i in range(N)]
    wr = [[Fr(v) for v in l] for l in base]; wf = [[float(v) for v in l] for l in base]
    dig = []
    for _ in range(3):
        wr = passo_vorticidade(wr, N, Fr(1, 10), Fr(1, 100), Fr(1), Fr(2), Fr(0))
        wf = passo_vorticidade(wf, N, 0.1, 0.01, 1.0, 2.0, 0.0)
        dig.append(max(len(str(v.denominator)) for l in wr for v in l))
    assert max(abs(float(a) - b) for la, lb in zip(wr, wf) for a, b in zip(la, lb)) < 1e-14
    assert dig[2] > dig[1] > dig[0]
