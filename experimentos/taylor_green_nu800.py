"""Taylor–Green em 128³ com nu = 1/800 (completa a série 1/400, 1/800, 1/1600)."""
import json, time
from via_sq.validacao import certificado
from via_sq.espectral import Fluxo
f = Fluxo(N=128, nu=1 / 800)
t0 = time.time()
U, h = f.roda(f.taylor_green(), T=10.0, dt=0.01, cada=20)
h["certificado"] = certificado(f, h); h["segundos"] = time.time() - t0
json.dump(h, open("tg800_128.json", "w"))
print("ok", h["certificado"], h["segundos"])
