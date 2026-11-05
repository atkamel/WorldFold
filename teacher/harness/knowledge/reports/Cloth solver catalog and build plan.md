# Cloth solvers we can build on, and how they fit our plan

*2026-10-16. Gap-check run: 3 finders (GitHub/web, academic, industry) and 1 completeness critic. It profiled about 100 solvers, about 85 of them new to our notes. Notes are in `research_notes/Cloth solver gap check/`. This report combines them with `reports/Real cloth catalog and error budget.md` (materials, data, error theory) and `reports/Realistic cloth simulators for robots.md` (Oct 3). Tags: ✅ verified from a primary source, 🟨 snippet or inference, ⬜ not found.*

## Bottom line

- **The most important find is a benchmark that already ran our step-2 experiment on our target engine.** It is Cloth-Var, "Same Action, Different Outcome" (IRI Barcelona, arXiv 2610.10801, 7 Oct 2026) ✅.
  - **Setup.** 269 real dynamic rollouts on 3 cloths. The measured gripper poses were replayed at 100 Hz as scripted held points in 4 calibrated engines.
  - **Held-out marker error:**

    | Engine | Error (mm) |
    |---|---|
    | **Clothilde** | **37–48** |
    | Isaac Lab | 70–78 |
    | Genesis | 82–90 |
    | **Newton VBD** | **79–94** |

  - **Newton's run-to-run spread doesn't follow the real spread** (rank correlation 0.01). It overstates the real spread 2.4×, and Newton was non-deterministic.
  - **Caveats.** The authors are the same group that makes Clothilde. Their calibration was a coarse grid plus one-parameter-at-a-time tuning, not a joint fit like our arm's CMA-ES.
- **It also gives a real repeat floor:** "same action, different outcome" across 24 configurations. That corrects my earlier report, which said no such floor was published.
- **No new open solver beats the shortlist,** but two are new to it:
  - **clothilde-sim** (Coltraro, Apache-2.0, Python): built-in scripted pins, air drag, static/dynamic friction. Validated at about 1 cm on mocap, and best in Cloth-Var.
  - **STARK** (Apache-2.0): IPC contact plus robot coupling, CPU only, hours per run.
- **ZOZO correction:** it *does* have air drag and lift (`air-density`, `air-friction`, `wind`) ✅. I told you it had none, from a docs page that didn't list them.
  - Its friction is smoothed (`friction-eps` 1e-5), so there's no exact stick ✅.
  - Moving pins along a path is fully exposed only in the Blender UI. The Python API exposes `move_by` ✅.
- **Exact stick friction exists in open code only as offline references.** ProjectiveFriction is GPL-3.0+; Argus is non-commercial. No open GPU solver with exact stick was found ⬜.

---

## 1. Shortlist by role in "build big, then strip"

### Reference tier: accurate, slow, and scored against real data, not trusted blindly

| Solver | Why | Limits | Licence | Tag |
|---|---|---|---|---|
| **clothilde-sim** (IRI) | `simulate(u, control)` pins nodes along any path, which is exactly our step-2 method. Air drag parameter. Floor and self friction. ~1 cm vs mocap; best in Cloth-Var | Nearly unstretchable and the same in every direction, so it suits woven, denim and terry, **not stretchy jersey knit**. CPU, no published speed. Its makers also ran Cloth-Var | Apache-2.0 | ✅ |
| **ZOZO ppf-contact-solver** | GPU, never passes through itself, air drag and lift, separate warp/weft bending, strain limits. Runs on our L4 | Smoothed friction. Stretch is the same in every direction. Path-scripted pins from Python are limited (`move_by`). "Not production ready" | Apache-2.0 | ✅ |
| **STARK** | IPC contact, robot and rigid coupling, animated boundary conditions. Reproduced 16/16 robot–towel lab outcomes | CPU, ~6 h for 16 towel runs on a 32-core machine. Smoothed friction | Apache-2.0 | ✅ |
| **libuipc** | GPU IPC with robot coupling, pip install | Smoothed friction; we write our own control | Apache-2.0 | ✅ |
| **ProjectiveFriction** (Inria) | **Exact Coulomb stick** | C++, CPU. GPL is fine for an internal tool, but not to ship | GPL-3.0+ | ✅ |

### Target tier: fast, where we strip down to

| Solver | Notes | Tag |
|---|---|---|
| **Newton** VBD / Style3D / XPBD, plus the OGC contact model | **v1.7.0 is out (2026-10-14); we run 1.6.1.** OGC is penetration-free, ~6–12 ms per step at 246k vertices on an RTX 4090 (from its paper teaser). Cloth-Var scored VBD at 79–94 mm, see above | ✅ |
| Newton templates | `kywind/cloth_sim` (VBD/Style3D cloth plus MuJoCo arms; **no licence**, read-only). Utah `clothcap` (MIT, fits 4D garment scans in Newton). RoboSimStudio (Newton real-to-sim) | ✅ |
| MuJoCo flex | 3.5 added `flexvert` constraints for coarser cloth meshes; mujoco is now 3.16 | ✅ |

### Not for us, and why

| Option | Reason |
|---|---|
| XRTailor | Apache-2.0, CUDA, but no Python API, no published speed, pins can only be fixed |
| Chrono | CPU, no cloth validation |
| NvCloth, Unreal Chaos, Obi, Jolt, Blender | Game or DCC tools. None checked against real cloth. Jolt has no cloth-on-cloth collision |
| ovphysx (standalone PhysX) | No static friction; no kinematic cloth points |
| Fashion tools: CLO / Marvelous Designer, Browzwear | Closed, no API to move pins along a path. Static drape only: CLO about 11% and Browzwear about 25% drape-coefficient error, against about 2% real repeat variation (86 fabrics, SEDDI) ✅ |
| Neural garment sims (HOOD, ContourCraft) | Driven by body poses; collision handling is learned, not guaranteed |

## 2. Licence traps, which matter for a startup

| Item | Problem |
|---|---|
| ARCSim and everything built on it (Argus, HYLC, DiffSim, GPU ARCSim) | Non-commercial |
| GarmentCode's Warp fork | Non-commercial |
| Wang 2011 data | Non-commercial |
| **FrankenTextiles** | MIT, plus a rider that bars giving it to OpenAI or Anthropic. **Don't use it through Claude** |
| YarnBall, ProjectiveFriction | GPL: fine for internal use, not for shipping a product |
| Coltraro data | Licence conflict (BY-SA vs BY-NC-SA) |
| kywind/cloth_sim, SIM1 | No licence file |

**Safe to build on:** Newton, ZOZO, clothilde-sim, STARK, libuipc, Gaia/OGC, BS-Cloth, ipc-toolkit (MIT), libshell (MPL-2.0).

## 3. New data leads

| Source | What | Status |
|---|---|---|
| **Cloth-Var** | 269 real rollouts, gripper poses at 100 Hz, 19 OptiTrack markers per cloth, measured μ. Wool 400, satin 100 and chiffon 76 g/m² | CC-BY; **only a sample episode is out so far** ✅ |
| Fabric-101 | 101 fabrics with loading/unloading curves | "Coming soon" 🟨 |
| Sperl 2022 | Production knit swatch tests | Linked, not opened 🟨 |
| U3M 1.1 `physics_1_1.json` | Raw fabric-tester force–distance curves in the length, width and bias directions | Open format ✅ |
| Utah clothcap | 4D garment scans at 1 mm (shape only) | MIT ✅ |
| Romero et al. 2021, "Physical validation of simulators" | Some popular shell and contact simulators failed simple physical tests | Paper behind a bot wall; data linked 🟨 |

**Still missing:** motion data on **cotton jersey and terry**, our top two materials. Cloth-Var used wool, satin and chiffon; Coltraro used cotton, denim, wool and polyester wovens. So our own measurements (option B) stay necessary for the T-shirt and towel.

## 4. How it all combines: the build plan

1. **Reference first, on the laptop ($0).** Run clothilde-sim on Coltraro's mocap and the Cloth-Var sample, with scripted pins. Reproduce their ~1 cm and 37–48 mm numbers and compute the floors. This is the cloth version of re-running NVIDIA's arm fit before trusting ours.
2. **Target on the L4.** Run the same replays in Newton 1.7 (VBD, Style3D, OGC contact) and fit jointly with CMA-ES, as we did for the arm. **First question:** can a proper fit beat Cloth-Var's 79–94 mm, and is Newton's non-determinism fixable (seeds, iterations)?
3. **Second reference for knits and stacked layers: ZOZO on the L4.** Clothilde can't model stretchy jersey, so ZOZO covers what it can't.
4. **Patch Newton only where residuals point:** air drag, the nonlinear knit stretch curve, friction. If exact stick turns out to matter, use ProjectiveFriction to measure how much.
5. **Contact (step 3) later:** STARK or libuipc as the reference, plus the two small real tests (cloth on the jaw material on an incline; SO-101 pinch-and-pull slip).
6. **Our own measurements (option B)** for jersey and terry, since no public motion data covers them.

## 5. Startup note

- **Fashion already pays for measured fabric:** Vizoo physX, SEDDI, the CLO and Style3D kits. They sell *static* properties only (drape, bending, stretch) ✅.
- **No seller of dynamic or contact cloth data for robotics was found** ⬜.
- **Cloth-Var shows academic groups are starting to build exactly this kind of benchmark.** The window is open, but not empty.

## Unverified and gaps

**Cloth-Var:**
- Only the abstract-level results and the critic's reading were used.
- The full dataset isn't released, and the real spread in mm wasn't extracted.

**Not checked:**
- **clothilde-sim:** speed and Windows install.
- **ZOZO:** the JupyterLab frontend's pin-path API.
- **OGC:** its friction model.
- **Newton 1.7:** its changes since 1.6.1.

**Code promised but not out:** Barrier-free AL, AGIPC (its repo returns 404), Mil2, JGS2, FLASH, RGBench GarmentDynamics.

**Not fully covered:**
- PyPI couldn't be keyword-swept (bot wall).
- Chinese, Japanese and Korean industry was covered by search only.
- The Semantic Scholar citation list for ZOZO's paper looks incomplete.

**Sources:** `s1_github_web.md`, `s2_academic.md`, `s3_industry.md`, `critic.md` in `research_notes/Cloth solver gap check/`.
