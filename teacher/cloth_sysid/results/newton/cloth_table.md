# Cloth step 2: Newton VBD fitted to real cloth

Held-out session (32 `*_hands` recordings, scored once after sha256 freeze `7d94cb70fccdd07a`), two corners scripted along their recorded paths.

| Model | paper metric (lit_marker, mean) | marker RMS mm (median) |
|---|---|---|
| rigid sheet carried by the corners | 1.183 | 20.4 |
| Newton as we had it | 1.785 | 27.6 |
| clothilde, paper's per-fabric params | 0.778 | 15.7 |
| Newton fitted, bending free (full) | 0.584 | 13.2 |
| Newton fitted, bending in published woven range (full) | 0.615 | 13.8 |
| Newton hand-off model, stripped | 0.670 | 15.8 |

- Gate 1a (<= 1.5x clothilde): **PASS**; gate 1b (at least half of clothilde's gain over rigid): **PASS**.
- Hand-off model: physbend (bending limited to published woven range).

| Rung | settings | held lit_marker | RMS mm | copy-sim-s per wall-s, L4: 32 copies / 1024 copies | NaN |
|---|---|---|---|---|---|
| full | sub 10, it 10, refine 2 | 0.584 | 13.2 | 39 / 280 | 0 |
| sub20 | sub 20, it 10, refine 2 | 0.584 | 13.0 | 10 / 136 | 0 |
| it5 | sub 10, it 5, refine 2 | 0.591 | 13.2 | 36 / 544 | 0 |
| sub5 | sub 5, it 10, refine 2 | 0.584 | 13.2 | 37 / 547 | 0 |
| sub5_it5 | sub 5, it 5, refine 2 | 0.585 | 13.0 | 72 / 1095 | 0 |
| sub2_it5 | sub 2, it 5, refine 2 | 0.645 | 15.0 | 179 / 2777 | 0 |
| sub5_it2 | sub 5, it 2, refine 2 | 0.595 | 13.4 | 167 / 2735 | 0 |
| refine1_sub5_it5 | sub 5, it 5, refine 1 | 0.640 | 14.8 | 74 / 3213 | 0 |

Gate 2 (fastest rung within 25 % of full): **sub2_it5**.
