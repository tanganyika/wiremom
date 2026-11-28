"""20 m quarter-wave vertical over average ground, 16 radials (1 mm copper, bare
or 0.75 mm PVC) above, resting on and in the soil: impedance and efficiency from
the 2-D near field corrected onto bare radials 3 mm above and 3 mm below the
surface (both, as a check).  Needs wiremom with extra_p-experiment.patch and the
eff-runner built from it:  python3 resting_antenna.py path/to/eff"""
import math, subprocess, sys, json
from xsec import solve
from effective_height import ES
EFF = sys.argv[1]
a, t_pvc, eps_pvc = 0.5e-3, 0.75e-3, 3.0
refs = {z: solve(a, 0, 1, z, ES)[0] for z in (3e-3, -3e-3)}

def ant(z_ref, dp, length):
    r = subprocess.run([EFF, str(z_ref), str(dp.real), str(dp.imag), str(length)], capture_output=True, text=True).stdout.split()
    return complex(float(r[0]), float(r[1])), float(r[2])

rows = []
for label, t in (("bare", 0.0), ("PVC", t_pvc)):
    ro = a + t
    cases = [("2 mm above", ro + 2e-3), ("1 mm above", ro + 1e-3), ("0.25 mm above", ro + 0.25e-3)]
    cases += [(f"resting, {f:.0%} in soil", ro * math.cos(math.pi * f)) for f in (0.05, 0.15, 0.20, 0.25, 0.30, 0.50)]
    cases += [("just under (0.25 mm)", -(ro + 0.25e-3)), ("1 cm deep", -1e-2)]
    for name, zc in cases:
        p = solve(a, t, eps_pvc, zc, ES)[0]
        row = {"wire": label, "case": name}
        line = f"{label:4} {name:24}"
        for length in (3.0, 5.0):
            (zu, eu), (zd, ed) = ant(3e-3, p - refs[3e-3], length), ant(-3e-3, p - refs[-3e-3], length)
            row[f"{length:.0f}m"] = {"z_up": [zu.real, zu.imag], "eff_up": eu, "z_dn": [zd.real, zd.imag], "eff_dn": ed}
            line += f"  {length:.0f} m: {zu.real:5.1f}{zu.imag:+6.1f}j {100*eu:4.1f}% / {zd.real:5.1f}{zd.imag:+6.1f}j {100*ed:4.1f}%"
        print(line, flush=True)
        rows.append(row)
json.dump(rows, open("resting_antenna.json", "w"), indent=1)
