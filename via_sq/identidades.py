"""Identidades exatas da Via dos Quadrados Perfeitos (verificação simbólica).

Cada função devolve uma expressão que deve simplificar para zero, ou um objeto
exato (polinômio mínimo, expansão) a ser comparado com o enunciado do artigo.
"""
import sympy as sp

I = sp.I
x, t, th = sp.symbols("x t theta")
DELTA = 1 + sp.sqrt(2)
PHI = (1 + sp.sqrt(5)) / 2


# ---------------- Parte II: Euler sem pi
def quarto_de_volta():
    return sp.expand(I ** 2 + 1)                                   # i^2 + 1 = 0


def diagonal_do_quadrado():
    return sp.simplify(((1 + I) / sp.sqrt(2)) ** 4 + 1)             # ((1+i)/√2)^4 + 1 = 0


def cossecantes_que_fecham():
    """(cossecante, passos) tais que n passos somam meia-volta."""
    casos = [(sp.Integer(1), 2), (2 / sp.sqrt(3), 3), (sp.sqrt(2), 4), (sp.Integer(2), 6), (sp.sqrt(5) + 1, 10)]
    out = []
    for c, n in casos:
        s = 1 / c
        z = sp.sqrt(1 - s ** 2) + I * s
        out.append((c, n, sp.simplify(sp.expand(z ** n) + 1)))
    return out


def passo_raiz14():
    return (sp.sqrt(13) + I) / sp.sqrt(14)


def polinomio_minimo_raiz14():
    return sp.minimal_polynomial(passo_raiz14(), x)                # 7x^4 - 12x^2 + 7


def expansao_raiz13_mais_i_11():
    return sp.expand((sp.sqrt(13) + I) ** 11)                      # -550496√13 + 331616 i


def cossecante_de_fechamento():
    s = sp.im(expansao_raiz13_mais_i_11()) / sp.Integer(14) ** sp.Rational(11, 2)
    return sp.nsimplify(sp.radsimp(1 / s))                         # 16807√14/10363


def fechamento_raiz14():
    z = passo_raiz14()
    return sp.simplify(sp.expand(z ** 11 * (-sp.conjugate(z) ** 11)) + 1)


# ---------------- Parte III: proporção de prata
def prata_equacao():
    return sp.expand(DELTA ** 2 - 2 * DELTA - 1)


def pell(n_max=6):
    pares = []
    for n in range(1, n_max + 1):
        a, b = sp.expand(DELTA ** n).as_coeff_add()
        y = sp.expand(DELTA ** n - a) / sp.sqrt(2)
        pares.append((int(a), int(sp.simplify(y)), int(a) ** 2 - 2 * int(sp.simplify(y)) ** 2))
    return pares


def lucas_fibonacci(n_max=12):
    return [(sp.lucas(n), sp.fibonacci(n), sp.lucas(n) ** 2 - 5 * sp.fibonacci(n) ** 2) for n in range(1, n_max + 1)]


def passo_de_prata():
    d = DELTA
    return {
        "quadrado": sp.simplify(sp.expand((d + I) ** 2 - 2 * d * (1 + I))),
        "quarta": sp.simplify(sp.expand((d + I) ** 4 - 8 * d ** 2 * I)),
        "meia_volta": sp.simplify(sp.expand((d + I) ** 8 + (8 * d ** 2) ** 2)),
        "volta": sp.simplify(sp.expand((d + I) ** 16 - (8 * d ** 2) ** 4)),
        "modulo": sp.simplify(sp.expand(d ** 2 + 1 - (2 * d + 2))),
    }


def polinomio_minimo_prata():
    return sp.minimal_polynomial((DELTA + I) / sp.sqrt(2 * DELTA + 2), x)   # x^8 + 1


def tangente_22_5():
    return sp.simplify(sp.tan(sp.pi / 8) - 1 / DELTA)


# ---------------- 3.5: hipérbole = cossecante algébrica
def cossecante_algebrica():
    C = (t ** 2 + 1) / (2 * t)
    B = (t ** 2 - 1) / (2 * t)
    T = sp.tan(th / 2)
    return {
        "hiperbole": sp.simplify(C ** 2 - B ** 2 - 1),
        "C_igual_csc": sp.simplify(sp.expand_trig(C.subs(t, T) - 1 / sp.sin(th))),
        "B_igual_menos_cot": sp.simplify(sp.expand_trig((B.subs(t, T) + sp.cos(th) / sp.sin(th)).rewrite(sp.tan))),
        "m_igual_menos_cos": sp.simplify(sp.expand_trig((B / C).subs(t, T) + sp.cos(th))),
        "inclinacao": sp.simplify(sp.diff(C, t) / sp.diff(B, t) - (t ** 2 - 1) / (t ** 2 + 1)),
    }
