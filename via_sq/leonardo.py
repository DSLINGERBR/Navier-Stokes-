"""Modelos físicos de baixa fidelidade para os projetos aéreos de Leonardo.

Parafuso aéreo: teoria do elemento de pá combinada com momento (BEMT) para
um helicoide de uma volta completa. Ornitóptero: polar de arrasto de asa
batente em voo nivelado, com eficiência de batida.

Todos os valores de material e de eficiência são parâmetros explícitos.
"""
from dataclasses import dataclass, field
import math
import numpy as np

RHO = 1.225          # densidade do ar ao nível do mar, kg/m³
G = 9.81
PHI = (1 + 5 ** 0.5) / 2
DELTA = 1 + 2 ** 0.5

# Potência humana de referência (W)
HUMANO = {"contínuo (pessoa comum)": 75.0, "atleta, cerca de 1 hora": 250.0, "pico de poucos segundos": 1000.0}

MATERIAIS = {
    "renascença": {"vela_kg_m2": 0.30, "longarina_kg_m": 1.50, "cd0_asa": 0.060},
    "moderno":    {"vela_kg_m2": 0.03, "longarina_kg_m": 0.15, "cd0_asa": 0.020},
}
PILOTO_KG = 70.0


# ---------------------------------------------------------------- perfil
def coeficientes(alpha):
    """Vela fina: linear até 12°, depois placa plana descolada."""
    a_s = math.radians(12)
    if abs(alpha) <= a_s:
        cl = 2 * math.pi * alpha * 0.9
        cd = 0.02 + 0.6 * alpha ** 2
    else:
        cl = 2 * math.sin(alpha) * math.cos(alpha)
        cd = 0.02 + 2 * math.sin(alpha) ** 2
    return cl, cd


# ---------------------------------------------------------------- parafuso aéreo
@dataclass
class Parafuso:
    nome: str
    D: float            # diâmetro, m
    passo: float        # avanço axial de uma volta, m
    cubo: float         # raio do cubo / raio externo
    voltas: float = 1.0

    @property
    def R(self):
        return self.D / 2

    def angulo_passo(self, r):
        return math.atan(self.passo / (2 * math.pi * r))

    def area_vela(self, n=400):
        r = np.linspace(self.cubo * self.R, self.R, n)
        return self.voltas * np.trapezoid(np.sqrt((2 * math.pi * r) ** 2 + self.passo ** 2), r)

    def massa(self, material):
        m = MATERIAIS[material]
        L_helice = self.voltas * (math.hypot(2 * math.pi * self.R, self.passo) + math.hypot(2 * math.pi * self.cubo * self.R, self.passo))
        L_raios = 8 * self.R * (1 - self.cubo)
        L_mastro = self.passo * self.voltas + 1.0
        return m["vela_kg_m2"] * self.area_vela() + m["longarina_kg_m"] * (L_helice + L_raios + L_mastro)

    def bemt(self, omega=1.0, n=160):
        """Empuxo e potência em pairado para rotação omega (rad/s).
        O helicoide de uma volta cobre o anel inteiro: corda somada = 2πr·voltas."""
        r = np.linspace(self.cubo * self.R, self.R, n + 1)
        rc, dr = 0.5 * (r[1:] + r[:-1]), np.diff(r)
        T = Q = 0.0
        for ri, dri in zip(rc, dr):
            theta, c_tot, U_t = self.angulo_passo(ri), 2 * math.pi * ri * self.voltas, omega * ri

            def residuo(v):
                ph = math.atan2(v, U_t)
                cl, cd = coeficientes(theta - ph)
                U2 = U_t ** 2 + v ** 2
                dT_be = 0.5 * RHO * U2 * (cl * math.cos(ph) - cd * math.sin(ph)) * c_tot
                return 4 * math.pi * RHO * ri * v ** 2 - dT_be

            lo, hi = 0.0, 3 * U_t
            if residuo(lo) >= 0:
                v = 0.0
            else:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if residuo(mid) < 0 else (lo, mid)
                v = 0.5 * (lo + hi)
            ph = math.atan2(v, U_t)
            cl, cd = coeficientes(theta - ph)
            U2 = U_t ** 2 + v ** 2
            T += 0.5 * RHO * U2 * (cl * math.cos(ph) - cd * math.sin(ph)) * c_tot * dri
            Q += 0.5 * RHO * U2 * (cl * math.sin(ph) + cd * math.cos(ph)) * c_tot * ri * dri
        return T, Q * omega

    def pairado(self, material, piloto=True):
        """Rotação e potência de eixo necessárias para sustentar o próprio peso."""
        W = (self.massa(material) + (PILOTO_KG if piloto else 0)) * G
        T1, P1 = self.bemt(1.0)
        omega = math.sqrt(W / T1)
        P = P1 * omega ** 3
        A = math.pi * self.R ** 2
        P_ideal = math.sqrt(W ** 3 / (2 * RHO * A))
        return {"peso_N": W, "massa_kg": W / G, "omega_rpm": omega * 60 / (2 * math.pi),
                "velocidade_ponta_m_s": omega * self.R, "potencia_W": P,
                "potencia_ideal_W": P_ideal, "figura_merito": P_ideal / P}


def parafuso_aureo(D=4.8):
    return Parafuso("Parafuso áureo (referência de Leonardo)", D=D, passo=D / PHI, cubo=1 / PHI ** 2)


def parafuso_prata(D=4.8):
    return Parafuso("Parafuso de prata", D=D, passo=D / DELTA, cubo=1 / DELTA ** 2)


# ---------------------------------------------------------------- ornitóptero
@dataclass
class Ornitoptero:
    nome: str
    envergadura: float
    alongamento: float          # envergadura² / área
    eficiencia_batida: float = 0.6
    oswald: float = 0.85
    clmax: float = 1.2

    @property
    def area(self):
        return self.envergadura ** 2 / self.alongamento

    def massa(self, material):
        m = MATERIAIS[material]
        return m["vela_kg_m2"] * 2 * self.area + m["longarina_kg_m"] * 3 * self.envergadura

    def voo(self, material, piloto=True):
        """Potência mínima em voo nivelado e a velocidade em que ocorre."""
        W = (self.massa(material) + (PILOTO_KG if piloto else 0)) * G
        cd0 = MATERIAIS[material]["cd0_asa"]
        S, b = self.area, self.envergadura
        v_stall = math.sqrt(2 * W / (RHO * S * self.clmax))
        V = np.linspace(v_stall, v_stall * 4, 2000)
        P = (0.5 * RHO * V ** 3 * S * cd0 + 2 * W ** 2 / (RHO * V * math.pi * b ** 2 * self.oswald)) / self.eficiencia_batida
        i = int(np.argmin(P))
        return {"peso_N": W, "massa_kg": W / G, "area_m2": S, "v_stall_m_s": v_stall,
                "v_otima_m_s": float(V[i]), "potencia_W": float(P[i])}


def ornitoptero_aureo(b=10.0):
    return Ornitoptero("Ornitóptero áureo (referência de Leonardo)", envergadura=b, alongamento=PHI ** 2)


def ornitoptero_prata(b=10.0):
    return Ornitoptero("Ornitóptero de prata", envergadura=b, alongamento=DELTA ** 2)


# ---------------------------------------------------------------- busca da escala que voa
def menor_escala(fabrica, avalia, alvo_W, lo=2.0, hi=200.0):
    """Menor dimensão característica cuja potência fica abaixo do alvo (bisseção em log)."""
    def p(x):
        return avalia(fabrica(x))["potencia_W"]
    xs = np.geomspace(lo, hi, 60)
    ps = [p(x) for x in xs]
    ok = [x for x, pp in zip(xs, ps) if pp <= alvo_W]
    if not ok:
        return None, float(min(ps))
    j = list(xs).index(ok[0])
    a, b = (xs[j - 1] if j else lo), xs[j]
    for _ in range(40):
        m = math.sqrt(a * b)
        a, b = (m, b) if p(m) > alvo_W else (a, m)
    return float(b), alvo_W
