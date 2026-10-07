# A Via dos Quadrados Perfeitos e da Proporção de Prata

**Autor:** Davi Cançado Dutra Cota — Brasil, 2026.
© 2026 Davi Cançado Dutra Cota. Publicado para leitura e verificação pela comunidade científica.

**Licenças:** artigo sob [CC BY-NC-ND 4.0](artigo/LICENCA_ARTIGO.md); código e dados sob a [Licença de Verificação](LICENSE) — ler e executar para verificar os resultados; outros usos exigem autorização do autor.

- Artigo: [`artigo/A_Via_dos_Quadrados_Perfeitos_Davi_Cota.pdf`](artigo/A_Via_dos_Quadrados_Perfeitos_Davi_Cota.pdf)
- Código de verificação: pacote `via_sq` (abaixo).
- Documento elaborado com modelo da Tencent: [`artigo/Documento_Tencent_Via_SQ_corrigido.pdf`](artigo/Documento_Tencent_Via_SQ_corrigido.pdf) — princípios filosófico-matemáticos do autor; cálculos e redação feitos por modelo de IA da Tencent (“Hyfour”); polimento textual final por Claude. Documento distinto do artigo final; publicado para verificação. Onde divergem, valem os resultados testados do artigo final (Parte VII).

O assistente Claude (Anthropic) foi usado como ferramenta técnica nos cálculos, no código e na redação técnica; os princípios filosófico-matemáticos são do autor.

---


© 2026 Davi Cançado Dutra Cota. Todos os direitos reservados. Trabalho concebido e realizado no Brasil.

Código que acompanha o artigo *A Via dos Quadrados Perfeitos e da Proporção de Prata*.
Permite a qualquer pesquisador verificar as identidades exatas e reproduzir os resultados numéricos.

## Instalação
    python3 -m pip install -r requirements.txt

## Verificação rápida (cerca de 5 segundos)
    python3 -m pytest -v testes

Resultado esperado: 31 testes aprovados e 5 `xfailed`. Os `xfailed` são hipóteses
testadas e **não confirmadas** (artigo, Parte VII): cada teste descreve a hipótese, e a
falha é o resultado documentado (`strict=True` acusa se alguma passar a valer).

| Arquivo de teste | O que verifica |
|---|---|
| `testes/test_identidades.py` | Partes II e III: Euler sem π, Proposições 1 e 2, Pell, Lucas–Fibonacci, cossecante algébrica = cossecante (substituição t = tan(θ/2)) |
| `testes/test_modelos.py` | Leonardo, solucionador espectral (Beltrami, Euler, incompressibilidade), Burgers, diferenças finitas, aritmética exata |
| `testes/test_resultados_publicados.py` | Números das tabelas do artigo contra `dados/*.json` e as hipóteses não confirmadas |

## Módulos (`via_sq/`)
- `identidades.py` — identidades exatas em sympy.
- `leonardo.py` — parafuso aéreo (elemento de pá + momento) e ornitóptero (polar de asa batente).
- `espectral.py` — Navier–Stokes 3D pseudoespectral periódico, regra de 2/3, RK4; energia, enstrofia, faixa de analiticidade δ.
- `validacao.py` — soluções exatas de Beltrami, Taylor–Green Re = 1600, certificado de resolução (δ·k_max ≥ 2).
- `burgers.py` — equação de Burgers com fator integrante (calibração do diagnóstico).
- `diferencas_finitas.py` — esquema de diferenças finitas: versão do rascunho original e versão corrigida (passo parabólico, forma antissimétrica).
- `aritmetica_exata.py`, `aritmetica_intervalar.py` — ponto flutuante, racionais exatos e intervalos.

## Execuções completas (reproduzem `dados/`; horas em CPU comum)
    python3 -m experimentos.leonardo_e_ns_inicial          # Leonardo e varredura inicial
    python3 -m experimentos.convergencia_condicoes_iniciais 32 64 96 128
    python3 -m via_sq.validacao 32 64 128                   # Beltrami e Taylor–Green Re = 1600
    python3 -m experimentos.taylor_green_nu800             # Taylor–Green Re = 800 em 128³
    python3 -m experimentos.diferencas_finitas 32 48       # esquema de diferenças finitas
    python3 -m via_sq.burgers
    python3 -m via_sq.aritmetica_exata
    python3 -m via_sq.aritmetica_intervalar

Os arquivos JSON são gravados no diretório corrente. Os de referência estão em `dados/`.

## Limites
Os resultados de Navier–Stokes são numéricos e não constituem prova de existência e
suavidade globais; ver Parte VI do artigo. O caso Re = 1600 em 128³ não é certificado.
