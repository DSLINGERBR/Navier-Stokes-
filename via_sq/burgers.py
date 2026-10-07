"""Burgers 1D: u_t + u u_x = nu u_xx, u0 = sin x. Calibra o diagnóstico da transição
parábola (viscosidade) -> hipérbole (transporte). Sem viscosidade há choque em t = 1."""
import json, math
import numpy as np

def roda(nu, N, T=2.0):
    x = np.arange(N) * 2 * math.pi / N; k = np.fft.rfftfreq(N, 1 / N); kc = N // 3
    mask = k < kc; U = np.fft.rfft(np.sin(x))
    dt = 0.1 * 2 * math.pi / N; n = int(T / dt); dt = T / n
    E1 = np.exp(-nu * k ** 2 * dt / 2); E2 = E1 ** 2      # fator integrante: viscosidade exata
    def nl(U):
        u = np.fft.irfft(U, N); ux = np.fft.irfft(1j * k * U, N)
        return -np.fft.rfft(u * ux) * mask
    out = {"t": [], "delta": [], "grad_max": []}
    for m in range(n + 1):
        if m % max(1, n // 200) == 0:
            E = np.abs(U) ** 2; sel = (k >= 2) & (k < kc) & (E > 1e-26 * E.max())
            d = -np.polyfit(k[sel], np.log(E[sel]), 1)[0] / 2 if sel.sum() > 3 else float("nan")
            out["t"].append(m * dt); out["delta"].append(float(d))
            out["grad_max"].append(float(np.abs(np.fft.irfft(1j * k * U, N)).max()))
        if m == n: break
        a = nl(U); b = nl(E1 * (U + .5 * dt * a)); c = nl(E1 * U + .5 * dt * b); d = nl(E2 * U + dt * E1 * c)
        U = E2 * U + dt / 6 * (E2 * a + 2 * E1 * (b + c) + d)
        if not np.isfinite(U).all(): break
    return out

if __name__ == "__main__":
    R = {}
    R["inviscido"] = roda(0.0, 2048, T=0.98)
    for nu in (0.1, 0.05, 0.02, 0.01, 0.005):
        R[f"nu={nu}"] = roda(nu, 2048)
    json.dump(R, open("burgers.json", "w"))
    for k, v in R.items():
        d = [x for x in v["delta"] if np.isfinite(x)]
        print(k, "delta_min=%.4f" % min(d), "grad_max=%.1f" % max(v["grad_max"]))
