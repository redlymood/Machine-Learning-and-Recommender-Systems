"""Readme 4.1 fixture: does the readme 3.4 stop rule (L1 < 1e-6, p0 = v) land
within the self-check tolerance (|p_i - ref_i| < 1e-6, script.js:2908)?
Analysis only (4 nodes, float64 and float32-rounded), not the app's PageRank."""
import struct

EDGES = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
REF = {"uniform": [0.3725268513, 0.1958239118, 0.3941492369, 0.0375],
       "restart A": [0.4522328999, 0.1921989825, 0.3555681176, 0.0]}
out = [sum(1 for a, _ in EDGES if a == i) for i in range(4)]
f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]

for name, v in (("uniform", [0.25] * 4), ("restart A", [1.0, 0, 0, 0])):
    for label, rnd in (("float64", lambda x: x), ("float32", f32)):
        p, k = list(v), 0
        while k < 500:
            nxt = [(1 - 0.85) * v[j] for j in range(4)]
            for a, b in EDGES:
                nxt[b] += 0.85 * p[a] / out[a]
            nxt = [rnd(x) for x in nxt]
            res = sum(abs(x - y) for x, y in zip(nxt, p))
            p, k = nxt, k + 1
            if res < 1e-6:
                break
        err = max(abs(x - r) for x, r in zip(p, REF[name]))
        print(f"{name:9s} {label}: {k:3d} iterations, residual {res:.2e}, max |p - readme| {err:.2e}"
              f" -> {'within' if err < 1e-6 else 'OUTSIDE'} 1e-6")
