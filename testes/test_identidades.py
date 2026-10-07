"""Partes II e III do artigo: identidades exatas."""
import sympy as sp
from via_sq import identidades as idt

x = idt.x


def test_quarto_de_volta():
    assert idt.quarto_de_volta() == 0


def test_diagonal_do_quadrado():
    assert idt.diagonal_do_quadrado() == 0


def test_cossecantes_que_fecham():
    assert all(r == 0 for _, _, r in idt.cossecantes_que_fecham())


def test_proposicao_1_polinomio_minimo():
    assert sp.expand(idt.polinomio_minimo_raiz14() - (7 * x**4 - 12 * x**2 + 7)) == 0


def test_proposicao_1_nao_e_inteiro_algebrico():
    p = sp.Poly(idt.polinomio_minimo_raiz14(), x)
    monico = [c / p.LC() for c in p.all_coeffs()]
    assert not all(sp.Rational(c).q == 1 for c in monico)


def test_proposicao_1_nenhum_n_ate_1000():
    z = complex(idt.passo_raiz14().evalf(30))
    assert min(abs(z**n + 1) for n in range(1, 1001)) > 1e-4


def test_expansao_binomial():
    assert sp.simplify(idt.expansao_raiz13_mais_i_11() - (-550496 * sp.sqrt(13) + 331616 * sp.I)) == 0


def test_cossecante_de_fechamento():
    c = idt.cossecante_de_fechamento()
    assert sp.simplify(c - 16807 * sp.sqrt(14) / 10363) == 0
    assert abs(float(c) - 6.068323) < 1e-6
    assert sp.factorint(16807) == {7: 5} and sp.factorint(10363) == {43: 1, 241: 1}


def test_fechamento_raiz14():
    assert idt.fechamento_raiz14() == 0


def test_prata():
    assert idt.prata_equacao() == 0
    assert idt.tangente_22_5() == 0


def test_pell():
    assert idt.pell() == [(1, 1, -1), (3, 2, 1), (7, 5, -1), (17, 12, 1), (41, 29, -1), (99, 70, 1)]


def test_lucas_fibonacci():
    assert all(abs(r) == 4 for _, _, r in idt.lucas_fibonacci())


def test_proposicao_2():
    assert all(v == 0 for v in idt.passo_de_prata().values())


def test_polinomio_minimo_prata():
    assert sp.expand(idt.polinomio_minimo_prata() - (x**8 + 1)) == 0


def test_cossecante_algebrica_e_a_cossecante():
    assert all(v == 0 for v in idt.cossecante_algebrica().values())
