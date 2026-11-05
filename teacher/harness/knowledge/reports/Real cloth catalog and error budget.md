# Real everyday cloth: what to model, how to measure it, and how to separate sim error

*2026-10-16. 6 research agents and 2 adversarial checkers. Notes are in `research_notes/Real cloth catalog and error budget/`. Checker corrections are applied throughout. Tags: ✅ verified from a primary source, 🟨 snippet, chart-read or my inference, ⬜ not found.*

## Bottom line

- **What to model.** By weight, the world makes mostly polyester. By count, though, the things people fold are mostly cotton or cotton-rich. T-shirts are the largest garment category (10.4 B pieces traded in 2023, 75% cotton-classed). Then come sweatshirts, underwear and trousers/jeans. Towels are almost all cotton terry. Silk is 0.1% of fibre. Model these first:
  1. Cotton-rich single jersey (T-shirts and LeHome tops)
  2. Cotton stretch denim (pants)
  3. Cotton terry (towels)
  4. Polyester knit (the second T-shirt type)
- **No public dataset gives us a ready-made "real garment + measured mechanics + real manipulation" set.** RGBench and Coltraro 2025 come closest. We have to measure our own garments. The tests are cheap: a kitchen scale, calipers, a ruler, a phone, an inclined board, or about INR 3,000 per fabric at a lab.
- **There is a standard theory for splitting error: ASME V&V 20 and V&V 40.** Re-read with it, the arm's leftover 0.284° is ~11× bigger than all the noise floors combined, so it is real model and parameter error.
- **Cloth sims have not yet shown the arm-style ladder** (floor < our fit < naive baseline). In the best head-to-head study, fitted simulators were 3–6× further from a real cloth than a *different* real cloth was.
- **Correction (same day, from the solver gap check).** A same-action repeat floor *has* been published: Cloth-Var (arXiv 2610.10801, 7 Oct 2026), with 269 real rollouts and real run-to-run spread across 24 configurations. It also scores Newton VBD at 79–94 mm held-out marker error. See `reports/Cloth solver catalog and build plan.md`. Blanco's and Gong's repeat takes remain $0 extra floors.

---

## Part 1: What the world buys

### Fibre by weight, 2025 (Textile Exchange, 2026 report)

| Fibre | All uses, Mt (share) | Apparel only, about 56 Mt |
|---|---|---|
| Polyester | 82.5 (59%) ✅ | ~54% 🟨 chart-read |
| Cotton | 25.7 (18%) ✅ | ~29% 🟨 |
| Viscose and other cellulosics | 7.2 (5%) ✅ | ~7% 🟨 |
| Nylon | 7.3 (5%) ✅ | ~4% 🟨 |
| Elastane | 1.6 (1.1%) ✅ | – |
| Wool | 1.0 (0.7%) ✅ | ~1% 🟨 |
| **Silk** | **0.09 (0.1%)** ✅ | – |

Only about 40% of all fibre goes into clothes. Textile Exchange calls its end-use model "first generation".

### Garments by count, 2023 world imports (UN Comtrade, reproduced exactly by the checker) ✅

| Garment | Pieces | Classed "cotton" |
|---|---|---|
| T-shirts and singlets (HS 6109) | 10.4 B | 75% |
| Knit sweaters and sweatshirts (6110) | 6.7 B | 52% |
| Briefs and underpants | 6.5 B | 61% |
| Woven trousers, including jeans | 5.3 B | 72% |
| Knit trousers | 4.1 B | – |
| Towels (6302, by weight only) | cotton terry 765 kt vs 107 kt man-made | – |

- "Cotton" in customs data means cotton is the largest fibre, so 60/40 and 50/50 blends count as cotton.
- These are trade figures. Goods made and sold inside China or India without being traded are missing.
- The median cotton T-shirt weighs about 0.20–0.21 kg (checker-corrected).
- The EU reference T-shirt (PEFCR v3.1) is 170 g: 70% cotton, 23% polyester, 6% viscose ✅. The reference pants are 450 g, 47% cotton, 4% elastane ✅.

### The fabrics behind those garments

| # | Material + construction | Garments | Weight, g/m² | Thickness | Tag |
|---|---|---|---|---|---|
| 1 | Cotton or cotton-rich single jersey | T-shirts (LeHome tops, our target) | 161–194 | 0.42–0.51 mm at ISO 1 kPa | ✅ |
| 2 | Polyester or polyester/elastane knit | sport T-shirts, dresses | ~120–220 | – | 🟨 |
| 3 | Cotton/polyester brushed fleece | hoodies, joggers | ~250–350 | not found | 🟨 |
| 4 | Cotton stretch denim, 3/1 twill, 1–2% elastane | jeans (LeHome pants) | 290–380 | ~0.5–0.85 mm 🟨 | ✅ |
| 5 | Cotton/elastane rib | underwear, fitted tops | ~160–280 | – | 🟨 |
| 6 | Cotton or poly/cotton poplin, percale | shirts, sheets | ~85–150 | – | 🟨 |
| 7 | Cotton terry (woven loop pile) | towels (our target) | 417–575, pile 5.4–9.0 mm | – | ✅ |
| 8 | Polyester/nylon taffeta shell | jackets | ~60–150 | – | 🟨 not a folding priority |

**Thickness depends on how hard you press.** One cotton jersey measured 0.92 mm on KES at 49 Pa. Other jerseys measured 0.42–0.51 mm on ISO 5084 at 1 kPa. Cloth lying on cloth presses at only about 2 Pa. So for contact, the low-pressure thickness is the one we need.

### Datasets with results on everyday garments

| Dataset | Garments | Fabric stated? | Real data | Use for us |
|---|---|---|---|---|
| **RGBench** (AAAI 2026) | 9 real garments, including a T-shirt and a shirt | Lab-measured (ASTM D1388/D3107). Per-garment configs in the MIT repo, e.g. `green_tshirt` density 220, friction 0.6, 0.255 kg ✅ (checker found these) | Point clouds, grasp/fling/fold, CC-BY, 7.1 GB ✅ | Closest "garment + params + motion" set. Fold Chamfer: GarmentDynamics 13.2 mm, Isaac 31.4 mm ✅ |
| **Coltraro** (IJRR 2025) | 4 flat fabrics: cotton, denim, wool, polyester | Measured density, friction 0.48–0.80, drape ✅ | 120 mocap recordings, with repeat takes ✅ | Best fit for an arm-style ladder. Fitted model mean error 0.5–0.7 cm, worst recording 1.15 cm ✅ |
| **Household Cloth Object Set** (2022) | 27 towels, rags, napkins, sheets; no T-shirts | Mass, size, stretch %; fibre only by name ✅ | Used by Blanco-Mulero 2024 and ICRA 2024 | Towel reference |
| ICRA 2024 Cloth Competition | 34 T-shirts and towels, 679 episodes | No ✅ | Yes, CC-BY, 83 GB | Unfolding results only (best coverage 0.60) |
| UniFolding | 60 real shirts | Fibre names only ✅ | Yes | Success 60–73% |
| LeHome real | 5 per category | **No** ✅ | 500 SO-101 episodes, Apache-2.0 | Our benchmark's own real data, but no materials |

None of the 31 datasets checked publishes measured mechanics *and* manipulation data for the same everyday garment, except RGBench's 9 garments.

---

## Part 2: How scientists measure cloth

### Standard tests, and what to do at home

| Property → sim parameter | Standard | At home | Repeatability |
|---|---|---|---|
| Weight per area → density | ISO 3801 / EN 12127 / ASTM D3776 | 5 squares of 10×10 cm on a 0.01 g scale; log the humidity | under 2% 🟨 my arithmetic |
| Thickness vs pressure → contact thickness, compression | ISO 5084 (1 kPa), FAST-1, KES-FB3 | 50.5 mm disc loaded with 10 g / 40 g / 204 g / 2 kg (49 Pa to 9.8 kPa), read with caliper depth | ⬜ not found |
| Stretch, warp/weft/bias → stretch curve, anisotropy | ISO 13934, ASTM D5034/5035; knits ASTM D2594, ISO 20932 | Hanging strips with 25 / 100 / 500 g, photographed on a mm grid | CV 1–2.4% ✅. D2594 between-lab precision is "poor" per ASTM ✅ |
| Shear → shear stiffness | **No fabric ISO/ASTM**; KES-FB1, FAST bias | 45° strip in the same rig, or a drawn grid | FAST bias CV 3.5% ✅ |
| Bending → bending stiffness | ASTM D1388 (cantilever, heart loop), FAST-2, KES-FB2 | Strip over an edge, photographed. **Fit the sim to the photo**; don't use the 41.5° formula, which may be biased up to 72% 🟨 | ⬜ |
| Drape → whole-shape check (held out) | ISO 9073-9 / Cusick | 30 cm circle on an 18 cm plate, overhead phone | Drape coefficient SD 0.8–3.2 points; wave count changes between repeats ✅ |
| Friction → μ per contact pair | **No fabric ISO/ASTM**; ASTM D1894 is for plastic film; KES-FB4 | Inclined board with the phone inclinometer. Pairs: cloth–cloth, cloth–table, **cloth–actual jaw material** | ⬜ |
| Damping and air → drag, virtual mass | **No standard** | Slow-motion video of a strip pendulum and a dropped square | Coltraro: air matters even at hand speeds ✅ |

**The standard loads mostly miss folding.** KES tensile runs up to 490 N/m, but a doll shirt carried by one pinch sees about 2–25 N/m 🟨. FAST (4.9–98 N/m) is the closest. So for knits, record the full low-load curve.

**Costs:**
- Clemson charges $57–$144 per test ✅. Bending, drape and friction aren't on its sheet.
- SITRA's full set is about INR 3,000 per fabric at non-member rates ✅.
- One KES module was quoted at JPY 6.6 M, and Kato itself says KES complies with no international standard ✅.
- Labs are best used to check our home rig on 2–3 fabrics.

### Measured datasets we can use now

| Dataset | What it has | Licence | Use |
|---|---|---|---|
| **Wang 2011** (ARCSim JSON) | 10 fabrics in SI units: density, a 6×4 nonlinear stretch table, 3×5 bending. Includes the 60/40 T-shirt interlock, 95/5 rib and 11 oz denim ✅ | UC research/non-profit | Fastest measured seed for T-shirt, rib and denim. The bending values are hinge coefficients; converting them to a physical bending stiffness isn't validated |
| **Feng 2022** | 618 fabrics: density plus bending warp/weft/bias. No fibre or name columns ✅ | No top-level licence | Randomization prior: the 190 fabrics at 0.14–0.25 kg/m² |
| **Gong 2024** | 25 fabrics on a Cusick drape tester. Fabrics 1–10 have 12 samples × 4 captures ✅. Fibre and construction are in the paper's supplement, not the repo | Unstated | A **real drape repeat floor**, no robot needed |
| Bouman 2013 | 30 fabrics, including fleece, taffeta and denim: bending and weight, plus wind videos ✅ | Citation request only | Only measured source for fleece |
| Coltraro 2025 | Mocap of 4 fabrics, with rep1–3 takes ✅ | BY-SA vs BY-NC-SA conflict | Dynamic fit and repeat floor |
| Blanco-Mulero 2024 | 3 rags × 3 takes per motion, Kinect point clouds ✅ | CC-BY | **Real repeat floor**, computable now |

**Coverage gaps** ⬜:
- No public bending, stretch or friction data for woven cotton terry or brushed fleece.
- No fabric-on-PLA, TPU or silicone friction anywhere.
- No dataset measures damping.
- Industry libraries (CLO, Style3D, Browzwear) are closed and use their own unit conversions.

---

## Part 3: Separating the error

### The standard framework

- **ASME V&V 20** ✅: E = S − D = δ_model + δ_num + δ_input − δ_D. The noise floors combine as u_val = √(u_num² + u_input² + u_D²).
  - If |E| ≫ u_val, the error is the model's fault.
  - If |E| ≲ u_val, the error is lost in noise.
  - V&V 20 is **not pass/fail**.
- **ASME V&V 40** ✅ (via the FDA's 2023 guidance) answers "is it small enough?". Set the tolerance **in advance**, tied to the decision the model supports.
  - Fitting data is weak evidence. Held-out data is the real test.
  - **Sim vs sim is not validation** (FDA wording) ✅.
- **Kennedy & O'Hagan 2001** ✅: you can't separate real-world randomness from measurement noise without **repeated trials**. Fitted parameters are best-fitting, not physical.
- **Brynjarsdóttir & O'Hagan 2014** ✅: fitted parameters and model error trade off against each other. Their example became *more* confident in the wrong value as data grew. Our arm's delay sitting at its 0 bound is exactly this.
- **Numerical error** ✅: use three mesh or timestep levels with refinement ratio > 1.3 (Roache's GCI). Oscillating results mean no valid GCI.

### The arm, re-read with V&V 20 (local files, checker-confirmed)

| Term | Size (deg) |
|---|---|
| E, held-out | **0.284** |
| Encoder quantization (u_D) | 0.025 |
| Timestep 2/4/5 ms (u_num); oscillating, so the spread is reported | 0.005 |
| Solver iterations 10/20 vs 20/40 | 0.0014 |
| GPU vs CPU, on the mean | 0.002 (≤0.021 per motion) |
| 1e-9 rad start nudges, on the mean | 0.003 (≤0.025 per motion) |
| **u_val** | **≥ ~0.026, about 11× below E** |

**Correction:** I had called 0.09° the arm's "chaos floor". It is the *worst single-instant* divergence, not an RMSE floor. On the mean, chaos is about 0.003°.

**The arm work did not measure:**
- real repeat scatter (no motion was repeated)
- systematic timing or zero offsets
- parameter uncertainty
- a fold-task tolerance

The arm verdict still stands: the remaining 0.28° is model and parameter error, not noise.

### Cloth floors found so far

| Floor | Value | Tag |
|---|---|---|
| Camera noise per pixel | Azure Kinect 0.6–0.9 mm at 0.8–1.5 m; RealSense flatness error D415 4.6 / D455 4.3 / L515 1.9 mm | ✅ |
| Chamfer floor from sensors alone | ~2–5 mm (Monte Carlo); Liu 2026 measured 5.2 mm with a ZED 2i | 🟨 / ✅ |
| Same cloth, same open-loop fold, 5 repeats | top-down IoU SD 0.01–0.11 (AdaFold) | ✅ |
| Same cloth, hand-driven repeats | marker Chamfer median 1.9 cm (0.2–5.8); start states already differ by 2.6 cm (Coltraro) | 🟨 computed |
| **Same cloth, same robot trajectory, repeats** | **⬜ not published.** Computable from Blanco's 3 takes per rag | ⬜ |

### Where published cloth sims sit relative to the floors

| Study | Sim error | Compared with | Ratio |
|---|---|---|---|
| Blanco-Mulero 2024, slow drag | fitted 7.2–10.2 cm | a *different* real rag: 1.8–2.4 cm | **3–6× worse** ✅ |
| Blanco-Mulero 2024, fling | 6.1–16.8 cm | different rag: 2.3–5.4 cm | 1.1–7× ✅ |
| Liu 2026 (FLASH), fold replay | 13.6 mm median | 5.2 mm measurement floor | ~2.6× ✅ |
| GAUGE 2026, satin, sim RMSE ÷ real scatter (20 trials) | stretching 0.73 (Isaac) / 1.51 (Genesis) / 0.73 (Newton); bending 7.9 / 11.7 / 19.9; fling 128 / 8.5 / 9.3 | real scatter = 1 | only stretching is under scatter ✅ |
| SimWeaver 2026 (Style3D co-authors) | Newton VBD scripted grasp-and-lift: 0% success, 77.5% penetration | – | interested party ✅ |

### Choosing a metric, the cloth version of "degree RMSE"

- **Chamfer distance is blind to in-plane slip.** Sliding the cloth 2 cm moves Chamfer by only about 0.4 mm (checker-corrected). Grasp slip is exactly our failure mode.
- **Primary:** tracked points, RMSE in mm.
- **Secondary:** Chamfer on visible surfaces only.
- **Task:** top-down IoU or our fold score.

### Proposed cloth ladder

Worst to best, each rung scored on held-out motions and garments:
1. Cloth glued to the gripper (null model)
2. Default sim parameters
3. Published parameters
4. Our measured parameters
5. Calibrated parameters

Every rung is reported next to four floors:
- sensor floor (static re-capture)
- real repeat floor (same cloth, same SO-101 trajectory, 5–10 takes)
- sim chaos floor (start nudges or seeds)
- numerics (mesh and timestep ladder)

**Showing the error is "marginal for folding"** (V&V 40):
1. Set tolerances in advance on grasp slip, corner position and fold-line position.
2. Show |E| + 2·u_val is under them.
3. The strongest form is fitted sim error ≤ the real repeat floor (GAUGE ratio ≤ 1).

No study ties sim error to real fold success under controlled conditions ⬜. Two findings point the same way:
- Real failures are mostly grasp failures, with about 1 cm of grasp-height margin (Matas 2018) ✅.
- Randomizing cloth physics significantly cut real fold error (Hietala 2022) ✅.

---

## What this means for choosing the cloth representation (evidence only)

**Keep:**
- **Friction with a real stick regime, and realistic thin contact.** Grasping dominates real failures. The earlier report found PBD has no stick state.
- **An air or damping term.** It matters at hand speeds (Coltraro).
- **Mesh edges ≤ 5 mm.** The sim's Chamfer floor is half its vertex spacing, and the bending length is about 1 cm.

**Probably strippable (one data point):**
- **Anisotropy.** It changed RGBench's fold error by only 2.6%.

**Measured values are seeds, not answers:**
- Blanco's fitted MuJoCo "mass" of 2.3–11.7 against real cloths of 0.05–0.10 kg ✅ shows fitted parameters are sim-specific.
- Vendor unit conversions are opaque.
- So: measure, then fit inside our chosen solver.

**Fits the "build big, strip later" plan:** Wang's tables are nonlinear and direction-dependent. Newton's Style3D solver takes direction-dependent stiffness, and VBD takes a single value per triangle. So Style3D can be the big model and VBD the stripped one.

## Options for the next step (Roy decides)

| Option | Cost | Needs |
|---|---|---|
| **A. Floors and a first rung from public data:** compute the real repeat floor from Blanco's 3 takes per rag and Gong's 12×4 drape repeats, then simulate Gong's drape test with Wang/Feng parameters | $0, laptop | nothing |
| **B. Measure our own garments:** the DIY kit above, on our T-shirt and towel | ~$0 | someone with the cloth, scale, calipers, phone |
| **C. Lab check of the home rig** on 2–3 fabrics | ~INR 3,000 per fabric (SITRA) or $57–144 per test (Clemson) | shipping swatches |

My recommendation is A now and B in parallel. This mirrors the arm, where public NVIDIA data came before our own arm.

## Corrections the checkers made

- **Comtrade garment weights:** median cotton T-shirt 0.214 kg, not 0.187; men's trousers 0.566 kg, not 0.510.
- **US T-shirt imports:** "largest US import category" is only true among categories counted in pieces.
- **RGBench:** per-garment parameters *are* public, in MIT-licensed configs.
- **Gong's metadata** is in the paper's supplement, not the repo.
- **Bouman's taffeta** averages 0.022 lbf·in² over its 3 samples.
- **Blanco's fitted Young's modulus** spans 1.1e4–8.5e4 Pa.
- **Coltraro:** "under 1 cm" holds on average only (A2 mean 0.71 cm, worst 1.15 cm).
- **FLASH** blames real failures on depth noise *and* self-occlusion.
- **Azure Kinect warm-up** is 40–60 min.
- **Chamfer slide blindness** is +0.4 mm, not 1.4 mm.
- **Arm "chaos floor":** 0.003° on the mean, not 0.09°.

## Unverified and gaps

**Not read in full:**
- Full ASME V&V 20, V&V 10 and V&V 40 texts (paywalled). These were read via a workshop procedure and the FDA reprint.
- Oberkampf & Roy, beyond the table of contents.

**Rough figures:**
- The apparel fibre split is chart-read (±~1 Mt).
- Jeans counts and the share of jeans with elastane are unsourced.

**No numbers found:**
- repeatability (r/R) for most ISO/ASTM fabric tests
- inter-lab reproducibility for KES/FAST
- DIY friction repeatability

**Never measured anywhere:**
- a same-cloth, same-robot-trajectory repeat floor
- a cloth divergence rate (Lyapunov)
- a mesh/timestep GCI for VBD or IPC cloth
- a controlled study of fold success vs sim error

**Licences:**
- Wang: non-profit only
- Coltraro: conflicting
- Gong and Bouman: unstated
- RGBench garment assets: non-commercial

**Sources:** six notes in `research_notes/Real cloth catalog and error budget/`: `market.md`, `garment_datasets.md`, `test_procedures.md`, `measured_datasets.md`, `error_theory.md`, `cloth_error_floors.md`. The earlier simulator survey is `reports/Realistic cloth simulators for robots.md`.
