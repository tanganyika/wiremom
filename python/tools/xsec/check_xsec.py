import math, cmath, sys
from xsec import solve
f = 14.15e6
eps0 = 8.8541878128e-12
es = complex(13, -0.005 / (2 * math.pi * f * eps0))
g0 = (es - 1) / (es + 1)
a = 0.5e-3
print("soil eps", es, "Gamma0", g0)
# 1. bare wire at two heights in air: p(h1) - p(h2) = Gamma0 ln(h1/h2) (+ O((a/2h)^2))
p5, n5 = solve(a, 0, 1, 5e-3, es)
p10, n10 = solve(a, 0, 1, 10e-3, es)
print(f"heights 5 / 10 mm: dp = {p10 - p5:.5f}, image theory {g0 * math.log(2):.5f}  ({n5}, {n10} triangles)")
# 2. insulated vs bare at 10 mm: dp = ln(b/a) (1/eps_i - 1)
t = 0.75e-3
pi10, _ = solve(a, t, 3.0, 10e-3, es)
print(f"insulation at 10 mm: dp = {pi10 - p10:.5f}, coax {math.log((a + t) / a) * (1 / 3 - 1):.5f}")
# 3. outer radius independence of a difference
p5b, _ = solve(a, 0, 1, 5e-3, es, R=3.0)
p10b, _ = solve(a, 0, 1, 10e-3, es, R=3.0)
print(f"R = 3 m: dp = {p10b - p5b:.5f}")
