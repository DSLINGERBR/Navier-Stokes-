"""Confere os números do artigo com os dados das execuções completas (pasta dados/).
Os testes marcados xfail(strict=True) registram hipóteses NÃO confirmadas: o teste
descreve a hipótese, e a falha é o resultado esperado e documentado."""
import pytest


def dmin(h, t0=0.5):
    return min(x for t, x in zip(h["t"], h["delta"]) if t >= t0)


def test_condicoes_iniciais_128(dados):
    ns = dados["ns_avanco"]
    esperado = {"Prata": (1.8754, 0.117), "Áureo": (3.2863, 0.105), "Taylor–Green": (2.1916, 0.082)}
    for k, (z, d) in esperado.items():
        h = next(v for kk, v in ns.items() if kk.startswith(k) and kk.endswith("N=128"))
        assert round(max(h["Z"]), 4) == z and round(dmin(h), 3) == d


def test_convergencia_96_128(dados):
    ns = dados["ns_avanco"]
    for c in ("Prata", "Áureo", "Taylor–Green"):
        z = [max(v["Z"]) for k, v in ns.items() if k.startswith(c) and (k.endswith("N=96") or k.endswith("N=128"))]
        assert abs(z[0] - z[1]) / z[1] < 0.002


def test_escala_de_kolmogorov(dados):
    casos = [(1 / 400, dados["ns_avanco"]["Taylor–Green | N=128"]), (1 / 800, dados["tg800_128"]),
             (1 / 1600, dados["validacao"]["TG1600 | N=128"])]
    prod = []
    for nu, h in casos:
        eps = 2 * nu * max(h["Z"]); keta = (eps / nu**3) ** 0.25
        prod.append(dmin(h) * keta)
    assert all(2.2 <= p <= 2.6 for p in prod)


def test_burgers_publicado(dados):
    b = dados["burgers"]
    for nu in (0.1, 0.05, 0.02, 0.01, 0.005):
        h = b[f"nu={nu}"]
        d = min(x for x in h["delta"] if x == x)
        assert d / nu >= 3.14159 and max(h["grad_max"]) < 1 / (2 * nu)


def test_certificado(dados):
    v = dados["validacao"]
    assert not v["TG1600 | N=32"]["certificado"]["resolvida"]
    assert not v["TG1600 | N=64"]["certificado"]["resolvida"]
    assert dados["tg800_128"]["certificado"]["resolvida"]


# ---- hipóteses não confirmadas (falha esperada e documentada)
@pytest.mark.xfail(strict=True, reason="Δt = h é instável para a difusão (Parte VII, item 2)")
def test_hipotese_passo_dt_igual_h_estavel(dados):
    assert all(c["estavel"] for c in dados["teste_racional"]["A"])


@pytest.mark.xfail(strict=True, reason="Laplaciano com pesos 1/i² não converge (Parte VII, item 3)")
def test_hipotese_laplaciano_com_pesos_converge(dados):
    e = [c["erro_laplaciano_pesos_1_k2"] for c in dados["teste_racional"]["B"]]
    assert e[-1] < e[0] / 4


@pytest.mark.xfail(strict=True, reason="a enstrofia cresce (Parte VII, item 4)")
def test_hipotese_derivadas_nunca_crescem(dados):
    c = dados["teste_parabola"]["convergencia_corrigido"]["48"]
    assert c["Zmax"] <= c["Z0"]


@pytest.mark.xfail(strict=True, reason="nenhum projeto voa com potência humana contínua (Parte IV.1)")
def test_hipotese_ornitoptero_voa_com_250W(dados):
    assert min(dados["resultados"]["aero"]["curvas"]["ornitoptero_prata_W"]) <= 250


@pytest.mark.xfail(strict=True, reason="a prata piorou o parafuso aéreo (Parte VII, item 6)")
def test_hipotese_parafuso_de_prata_melhor(dados):
    p = dados["resultados"]["aero"]["parafuso"]
    assert p["Parafuso de prata"]["moderno"]["potencia_W"] < p["Parafuso áureo (referência de Leonardo)"]["moderno"]["potencia_W"]
