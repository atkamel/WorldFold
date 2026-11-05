# Contact physics, not cloth stiffness, decides realism

As of October 2026, the most physically faithful way to simulate a garment being grasped and folded is **incremental potential contact (IPC)**: an implicit shell model with barrier contact that cannot penetrate, and smoothed Coulomb friction that includes a sticking regime. It is available today as **libuipc/pyuipc** (Apache-2.0, pip wheels for Windows and Linux) and through **Genesis's IPC coupler**, which couples the cloth and the robot both ways by default. Its cost is speed: the one public two-arm IPC shirt fold ran at about **1.1 s per frame on an RTX 5070 Ti**, and the final grasp still slid about 90 mm. The fastest realistic option a student team can adopt in weeks is **NVIDIA Newton's VBD cloth solver** (Apache-2.0, consumer GPU, URDF/USD import, a working Franka-folds-a-T-shirt example at about 30 FPS on an RTX 4090). It is still labelled experimental, uses a single smoothed friction coefficient, has no crease plasticity, and an independent benchmark calls it "overly stiff in bending". The simulators with the best published real-world results, FLASH (70% zero-shot T-shirt folding) and RGBench's GarmentDynamics, have no released code. Most of the team's symptoms are contact-model failures of PBD particle cloth rather than wrong material numbers: slip, creep, phasing and sticking on release all come from the contact model. The 147 kg shirt is a separate, trivially fixable error, because the particle mass is set per particle. The recommendation is to prototype the SO-101 with ClothesNet garments in standalone Newton VBD and check it against a few Genesis-IPC and real-fabric micro-tests. Moving the full pipeline to Isaac Lab 3 + Newton should wait until that passes. Meanwhile, the LeHome teacher should stay on patched PhysX particle cloth.

## Every symptom traces to PBD contact, and NVIDIA has already removed it

The team's environment runs a solver its vendor has abandoned. NVIDIA's migration guide states that "the PhysX particle cloth feature, its Omni PhysX integration, and the related schemas have been removed", replaced by surface deformables built on "a XPBD FEM corotational linear elasticity model", and warns that behaviour will not carry over one to one ([Omni Physics Deformable Migration Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/deformable_migration.html)). Isaac Lab's own maintainers said the Orbit paper's particle cloth "was quite difficult to tune for stable simulation" ([IsaacLab discussion #587](https://github.com/isaac-sim/IsaacLab/discussions/587)).

The symptoms map onto known properties of position-based friction and contact.

**Slip and creep.** Classic PBD friction "cannot model static friction because the frictional damping forces cannot correct for the position changes already made during the constraint solve. Consequently, particle positions drift" ([Macklin et al. 2014](https://mmacklin.com/uppfrta_preprint.pdf)). The team's cloth-on-cloth friction of about 0.3 is not unphysical in itself. Cotton single jersey measured against steel gives a kinetic μ of **0.29 to 0.31** dry ([Hes, Marmarali et al. 2008](https://dergipark.org.tr/tr/download/article-file/218065)). What PBD lacks is a stick state, so any sustained tangential load produces creep regardless of the coefficient.

**Phasing and no force on the robot.** NVIDIA forum users report that the Franka gripper "penetrates the cloth, causing the fabric to stick to the gripper when it opens". Staff advice was convex finger colliders, softer drives and 240 to 360 Hz stepping ([forum thread 332704](https://forums.developer.nvidia.com/t/some-critical-issues-in-isaac-4-5-particle-cloth-that-prevent-realistic-cloth-simulation/332704)). DexGarmentLab's authors wrote that PBD's "loosely connected particles often allow grippers to penetrate the garment without achieving effective lifting" ([arXiv 2505.11032](https://arxiv.org/html/2505.11032)).

**Sticking on release.** This is the same penetration problem seen from the other side. Particles trapped inside the jaw colliders, plus LeHome's adhesion term, keep the cloth on the fingers when they open.

**No creases.** The cause is spring-based bending with no rest-angle or plastic update.

**The 147 kg shirt.** This is arithmetic. LeHome's YAML sets `particle_mass: 1e-2` ([LeHome YAML](https://github.com/lehome-official/lehome-challenge/blob/a805ad2f7ab52a4583066fc4ee5180459a7f9d15/source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml)), and 0.01 kg × about 14,700 particles is 147 kg. That match settles the earlier open question: the value is per particle. A real doll shirt of about 0.2 to 0.3 m² of fabric at the T-shirt interlock density of **0.187 kg/m²** ([Wang, O'Brien, Ramamoorthi 2011](https://escholarship.org/content/qt7g35q2x7/qt7g35q2x7.pdf)) weighs roughly **40 to 55 g**, about 3,000 times less (the area is my estimate).

The pattern across the field confirms that friction grasping of thin simulated cloth is hard everywhere. Grippers still slip in MuJoCo flex ([mujoco #1480](https://github.com/google-deepmind/mujoco/issues/1480)) and in Genesis PBD ([genesis-world #199](https://github.com/Genesis-Embodied-AI/genesis-world/issues/199)). Most benchmarks sidestep the problem by pinning cloth to the gripper. SoftGym says it does so "to decouple the challenges in learning low-level grasping skills from high-level planning" ([arXiv 2011.07215](https://arxiv.org/html/2011.07215)), and FoldNet++ on Style3D FEM attaches nearby vertices when the gripper closes ([arXiv 2609.12433](https://arxiv.org/html/2609.12433)). The simulators that do hold cloth by friction share three ingredients: an implicit solve, friction defined as an energy or constraint with a real stick regime, and a finite contact thickness.

## Seventeen simulators sort into three tiers on contact fidelity

The table rates each candidate on the properties that matter for the team's problems. "Stick" means the friction model has a static regime rather than pure velocity damping. Speeds are the only published figures found, so they are not like-for-like.

| Simulator / solver | Friction model | Creases / plasticity | Two-way robot coupling | Penetration-free | Speed (published) | GPU | Robot import | Licence | Maturity |
|---|---|---|---|---|---|---|---|---|---|
| **libuipc / pyuipc** | Smoothed Coulomb, stick approximated below εᵥ, "scene-adaptive tolerances" | Shell bending, no plasticity listed | Yes: affine-body links in the same solve; articulation constraint added 2026-03 | **Yes** (barrier + CCD) | No standalone cloth figure | CUDA 12.8, sm 7.5 to 12.0 | URDF, OBJ, glTF | Apache-2.0 | Active, "v1.0 forthcoming" |
| **Genesis IPC coupler** (libuipc backend) | As above; default μ 0.5 | Bending stiffness parameter only | **Yes, on by default** | **Yes**; d̂ default 1 mm | **~1.13 s/frame**, 13.8k-face shirt, RTX 5070 Ti | Consumer GPU | URDF/MJCF (unverified) | Apache-2.0 (unverified) | Coupler docs moved; only third-party garment demo |
| **Newton VBD** | Single μ with `friction_epsilon` velocity smoothing | Dihedral bending, no plasticity documented | Shared collision pipeline with Featherstone/MuJoCo-Warp robot (coupling direction disputed, see below) | Log-barrier or penalty, not guaranteed | **~30 FPS** single scene RTX 4090; multi-env 0.42× PhysX | Maxwell+, driver 545+ | URDF, MJCF, USD | Apache-2.0 | "Experimental" API |
| Newton XPBD / Style3D | Not fetched / proprietary Pro | Style3D anisotropic | Style3DPro has hand-push example | Not documented | Not found | GPU | Via Newton | Apache / **licensed SDK** | Pro needs SynReal licence |
| PhysX surface deformables (Isaac Sim 5.x/6.x) | Dynamic only; "static friction is unsupported" | Bend stiffness + rest angles, no plasticity | Rigid contact "NOT fully supported" per one forum reply | No | Not found | NVIDIA GPU | Isaac USD | NVIDIA | Beta |
| PhysX PBD particle cloth (current) | Position-correction, no stick | None | Weak; fingers phase | No | Fast | NVIDIA GPU | Isaac USD | NVIDIA | **Removed** |
| Genesis PBD | Static + kinetic coefficients (default 0.15), positional | Bending compliance only | Yes, impulse-based | No | Fast; diverges multi-env | GPU | URDF/MJCF | Apache-2.0 (unverified) | Grasp slip issue closed via attachment API |
| MuJoCo flex / MJWarp | Soft-constraint Coulomb cone | Shell bending | Yes | No (soft constraints) | Fails at 40k vertices; 65× slower than GarmentDynamics at 20k | CPU; MJWarp flex "experimental" | MJCF/URDF | Not in notes | Grasp slip unresolved |
| C-IPC (Codim-IPC) | IPC smoothed Coulomb | Dihedral, strain limiting, no plasticity (unverified) | Only kinematic pinning in robotics use | **Yes**, with real shell thickness | **~14 min per fold** on 4-core CPU | CPU | None | Apache-2.0 | 5 commits, unmaintained |
| GIPC / StiffGIPC | Barrier friction approximation | Strain limiting, Wu & Kim bending | Not robot-oriented | Yes | StiffGIPC up to 10× GIPC | CUDA, sm_86 default | None | MPL-2.0 | Superseded by libuipc |
| ThinShellLab | IPC smooth friction, stick + slip | **Bending plasticity (creases persist)** | Custom manipulators | No (quadratic penalty) | 2.74 fps forward, RTX 4090 (ambiguous unit) | Taichi GPU/CPU | Write your own | Not shown | Research code, 19 commits |
| DiffCloth | **Dry Coulomb**, differentiable | Not extracted | Kinematic attachment | No (unverified) | CPU | CPU | None | MIT | Research code |
| SOFA | Constraint-based Coulomb | FEM | Via SofaPython3 scripting | No | No data | CPU | No robot models | Not in notes | Mature, manual setup |
| Bullet / PyBullet | Margin-based | Weak | Grippers "tunnelled ... or repelled" cloth | No | n/a | CPU | URDF | Not in notes | "Rudimentary and experimental" |
| ArcSim | Not robot-oriented | **Adaptive remeshing + bending plasticity** | None | No | n/a | CPU | None | Not in notes | Legacy |
| FLASH | **Signorini-Coulomb via Lagrange multipliers**, "accurate frictional sticking" | ARAP + bending | Bilateral grasp constraints in benchmark | Exact complementarity | **5.68 ms/step** single env, RTX 5090 | GPU | Custom | No code | Paper only |
| RGBench GarmentDynamics | Potential contact, friction fitted to measured roughness | Anisotropic FEM | Benchmark harness | No (untangling after the fact) | **2.7 ms/step at 40k vertices** | GPU | Custom | Code "will be released" | Paper + harness |

Sources for the table: libuipc ([README](https://github.com/spiriMirror/libuipc); [project site](https://spirimirror.github.io/libuipc-web/)); Genesis IPC ([coupler docs v0.3.14](https://genesis-world.readthedocs.io/en/v0.3.14/user_guide/advanced_topics/ipc_coupler.html); [shirt demo](https://github.com/happy1041/genesis-ipc-shirt-demo)); Newton ([README](https://github.com/newton-physics/newton); [SolverVBD API](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverVBD.html); [issue #4141](https://github.com/newton-physics/newton/issues/4141); [NVIDIA blog](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/)); Style3D ([SynReal/newton](https://github.com/SynReal/newton)); surface deformables ([Deformable Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/deformable_bodies.html); [forum 359023](https://forums.developer.nvidia.com/t/isaac-sim-5-1-ubuntu-thin-sheet-deformable-beta-deprecated-behaves-like-rigid-little-no-visible-deformation/359023)); Genesis PBD ([PBD Elastic docs](https://genesis-world.readthedocs.io/en/latest/api_reference/material/pbd/elastic.html)); MuJoCo ([MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/); [RGBench](https://arxiv.org/html/2511.06434v2)); C-IPC ([arXiv 2012.04457](https://arxiv.org/abs/2012.04457); [Codim-IPC](https://github.com/ipc-sim/Codim-IPC); [De Gusseme & wyffels 2022](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.989702/full)); GIPC ([GPU_IPC](https://github.com/KemengHuang/GPU_IPC)); StiffGIPC ([arXiv 2411.06224](https://arxiv.org/html/2411.06224)); ThinShellLab ([arXiv 2404.00451](https://arxiv.org/html/2404.00451)); DiffCloth ([GitHub](https://github.com/omegaiota/DiffCloth)); SOFA ([Soft Robotics Toolkit](https://softroboticstoolkit.com/sofa/tutorial/step-7)); Bullet ([Matas et al. 2018](https://arxiv.org/pdf/1806.07851)); ArcSim ([Berkeley](http://graphics.berkeley.edu/resources/ARCSim/)); FLASH ([arXiv 2604.17513](https://arxiv.org/html/2604.17513v1)).

The first tier, IPC through libuipc or Genesis, is the only group that combines guaranteed non-penetration, a stick regime and two-way coupling with a URDF robot. In the IPC Toolkit formulation, friction is lagged Coulomb with a smooth function f₀ that "approximates the switch from dynamic to static friction using a velocity threshold εᵥ" ([IPC Toolkit](https://ipctk.xyz/tutorials/getting_started.html)). Stiction is therefore approximated, and a tiny creep below εᵥ remains, but that creep is orders of magnitude smaller than PBD drift. The second tier is Newton VBD and the unreleased FLASH and RGBench solvers: implicit, fast and robot-aware, but with friction regularised (VBD) or code unavailable (FLASH, RGBench). The third tier is everything PBD- or penalty-based. Genesis PBD, MuJoCo flex, Bullet and the team's current solver all have public grasp-slip reports.

### Mapping the team's six problems to the realistic candidates

| Observed problem | PhysX PBD (now) | PhysX surface deformables | Newton VBD | Genesis IPC / libuipc | ThinShellLab |
|---|---|---|---|---|---|
| Slippery cloth / cloth-on-cloth μ 0.3 | No stick state | Dynamic friction only | Single μ, epsilon-smoothed; examples use robot μ 1.5 | **Stick regime**; demo used robot μ 2, still 90 mm slip | Stick + slip |
| No creases | Springs | Rest bend angles | Dihedral bending, no plasticity | Shell bending, no plasticity | **Plastic yield** |
| Creep when untouched | Drift by design | Unknown | Small creep possible below ε (inferred) | Creep below εᵥ only | Same as IPC |
| Fingers phase, no force back | Yes | Reported unreliable rigid contact | Barrier/penalty; two-way claimed | **Barrier; two-way default** | Penalty, can penetrate |
| Sticks on release | Penetration + adhesion | Unknown | No adhesion term (inferred) | No adhesion; force vanishes past d̂ (inferred) | Unknown |
| 147 kg mass | Per-particle setting | Density-based | Set per particle/area | Density ρ per material | Density |

No candidate fixes creases except ThinShellLab and ArcSim, and neither has robot-arm support. For a folding task this matters less than it seems. A fold held flat by its own weight and IPC-grade friction stays folded without plasticity. Only the permanent set of a pressed cotton crease is lost, and there is no measurement of how much that set affects folding success at doll scale.

## Published sim-to-real tests measure cloth shape, not grasp physics

Every head-to-head benchmark found compares cloth *geometry* against point clouds. None measures hold force, slip distance or release, which are the team's actual failures.

The cleanest comparison is Blanco-Mulero et al. (RA-L 2024). They fitted MuJoCo, Bullet, Flex (the PBD lineage inside Isaac) and SOFA to real 50 × 70 cm cloths with 500 rounds of Bayesian optimisation and CMA-ES. On the dynamic fling, MuJoCo and SOFA reached a Chamfer distance of about **0.07 to 0.08 m**, while Flex reached **0.16 to 0.17 m**. That is "two times lower than those in Bullet and Flex". In the quasi-static drag all four engines landed between 0.072 and 0.102 m ([arXiv 2310.09543](https://arxiv.org/pdf/2310.09543)). The authors add two warnings: "all engines need to be fine-tuned for the specific simulation frequency", and T-shirts remain future work.

RGBench reports its GarmentDynamics FEM solver at a Chamfer distance of **0.0132 m versus 0.0314 m for Isaac Sim** on T-shirt folding, at 2.7 versus 26.6 ms per step for 40k vertices ([arXiv 2511.06434](https://arxiv.org/html/2511.06434v2)). This is an interested party's result. GAUGE tested Isaac Sim, Genesis and Newton across 14 task families and found "no uniformly faithful physics engine", with "rapid textile motion" among the weak spots ([arXiv 2608.05948](https://arxiv.org/abs/2608.05948)). Ru et al. found anisotropic FEM beats isotropic MPM. They also found real-to-sim fitting works for folding (Chamfer distance 0.015) but "shows limitations in dynamic scenarios" ([arXiv 2503.16310](https://arxiv.org/html/2503.16310)).

Task-level transfer is thinner but more relevant. FLASH reports **zero-shot real folding of 85.8% for towels, 60% for shorts and 70% for T-shirts** on two robot arms. In its benchmark, Isaac's FEM/XPBD cloth showed "excessive wrinkling", Genesis PBD showed "large elastic snapping and poor frictional stability", and Newton VBD was "overly stiff in bending" ([FLASH](https://arxiv.org/html/2604.17513v1)). A custom MPM thin-shell simulator reached **81.3% zero-shot** on garment reversal ([Right-Side-Out](https://arxiv.org/abs/2509.15953)). ThinShellLab deployed one task on a real robot with a small reported gap ([arXiv 2404.00451](https://arxiv.org/html/2404.00451)). **No published study validates IPC or VBD cloth against a real robot grasp.** The first robotics use of C-IPC replaced the gripper with pinned vertices and had no real-robot test ([De Gusseme & wyffels 2022](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.989702/full)).

Three conclusions follow. For slow, quasi-static folding, **parameter identification and timestep matter more than the engine**. The PBD/Flex lineage is the documented worst case on dynamic motion. And the claim that IPC is "most realistic" for grasping rests on its formulation, not on measured robot data. A team that runs even a crude real hold-and-slip test would be producing evidence the literature lacks.

## Realistic jersey parameters, and why doll scale makes cloth look stiffer

The measured starting point for a cotton or cotton-blend T-shirt knit is set out below.

**Areal density.** About 0.15 to 0.23 kg/m². Wang 2011 measured its gray interlock T-shirt fabric at 0.187 kg/m² ([Wang et al. 2011](https://escholarship.org/content/qt7g35q2x7/qt7g35q2x7.pdf)). Ejaz et al. list 100% cotton jersey at 229 g/m² and 50/50 jersey at 185 g/m² ([Ejaz et al. 2023](https://www.paspk.org/wp-content/uploads/2024/03/ES-822.pdf)).

**Bending rigidity.** A patent summary gives the KES range for cotton knits as 0.002 to 0.05 gf·cm²/cm ([USPTO 5427156](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5427156)), which converts to roughly **0.2 to 5 µN·m**. This value is snippet-level. Knits are strongly anisotropic, with course direction much softer than wale.

**Friction against hard surfaces.** Jersey on steel is about 0.29 to 0.36, rising with moisture ([Hes et al. 2008](https://dergipark.org.tr/tr/download/article-file/218065)). Cotton against silicone gave the highest friction of four surfaces tested ([Tribology Letters 2021](https://link.springer.com/article/10.1007/s11249-021-01502-1)), but the number was not retrieved.

**Working values for other contacts.** Reasonable starting values are fabric-on-fabric 0.3 to 0.6, jaw plastic 0.2 to 0.35 and silicone pads 0.6 to 1.0. These are interpolations from the ranges above, not measurements.

**Stretch.** No verified jersey modulus was found. The often-quoted cotton value of E = 0.821 GPa with 0.318 mm thickness is for **woven** cotton ([De Gusseme & wyffels 2022](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.989702/full)) and badly overstates knit stiffness at low load.

### Setting these values in a simulator

**Do not derive bending from E and thickness.** Shell simulators take E, ν and thickness h, so membrane stiffness is about E·h and plate bending is about E·h³/12(1−ν²). For the woven cotton values that formula gives about 2,300 µN·m, which is **500 to 10,000 times** measured knit bending. Fabric bends by fibre slip, not as a solid plate. Set the bending stiffness separately from a measured B. Where the API only takes E, ν and h (the Genesis IPC coupler exposes a separate `bending_stiffness`), use a separate bending parameter.

**XPBD compliance.** In XPBD solvers, compliance is the inverse of stiffness, in m²/N ([Macklin](https://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/)).

**Treat measured values as seeds, not answers.** Fitted parameters are specific to the simulator, the mesh resolution and the timestep. Feng et al. deliberately estimate "simulation" rather than physical parameters for this reason ([Feng et al. 2022](https://wanghmin.github.io/publication/feng-2022-lbb/)). Blanco-Mulero's frequency sensitivity says the same thing.

**Garment mass.** With a 0.187 kg/m² fabric and about 14.7k particles, per-particle mass is about **3 to 4 × 10⁻⁶ kg**, not 10⁻².

### Doll-scale caveats

The bending length C = (B / w·g)^⅓ is a material property and does not shrink with the garment. With B = 2 µN·m and w = 0.18 kg/m² it is about **1 cm**. An adult shirt spans about 70 bending lengths, while a 30 to 45 cm doll shirt spans 30 to 45, so **real jersey at doll scale drapes visibly stiffer, with fewer and larger wrinkles**. If the real doll garment is real jersey, simulate the real B. Do not scale it down to mimic an adult shirt.

**Mesh resolution.** Edges must stay well under C, about 5 mm or less. A doll shirt with 10 to 15k vertices has edges of roughly 4 to 5 mm (estimated), which is just adequate.

**Hems, seams and collars.** These are proportionally larger on a small garment and will likely need local stiffening. No source tests this.

**Units and contact gaps.** Newton's `cloth_franka` runs "in centimeter scale for numerical stability of the VBD solver" ([example](https://raw.githubusercontent.com/newton-physics/newton/main/newton/examples/cloth/example_cloth_franka.py)), so doll garments should be simulated in centimetres too. IPC's contact gap is also a problem at this scale. The Genesis demo found d̂ = 1.0 mm unstable and used 1.5 mm ([demo](https://github.com/happy1041/genesis-ipc-shirt-demo)). A four-layer doll fold would then read about 6 mm thick, which is visibly thicker than real jersey.

### Measuring the team's own fabric for under an hour of work

**Areal density.** Cut a 10 × 10 cm square of the real doll shirt fabric, or weigh the whole shirt and estimate its area.

**Bending.** Run the ASTM D1388 cantilever test. Slide a 2.5 cm strip over a table edge until its tip droops to 41.5°. The bending length C is half the overhang, and B (µN·m) = W (g/m²) × C³ (mm) × 9.807 × 10⁻⁶ ([Ejaz et al. 2023](https://www.paspk.org/wp-content/uploads/2024/03/ES-822.pdf), formula verified). Do this in both the wale and course directions.

**Stretch.** Hang weights from a strip and photograph it against a ruler, as a cheap version of Wang 2011's rig.

**Friction.** Use an inclined plane: tan θ at the onset of sliding gives the static μ. Test fabric on fabric, on the jaw material and on the table.

**Fitting.** Record two phone videos, one of a corner lift-and-drop and one of a slow half-fold. Then fit damping and friction in the chosen simulator with Bayesian optimisation or CMA-ES on Chamfer distance, as Blanco-Mulero did.

## Recommendation: prototype in Newton, referee with IPC, keep LeHome for the teacher

### Primary path: standalone Newton VBD, judged by a go/no-go test

The most realistic simulator the team can adopt within weeks, on consumer or rented GPUs, while keeping the SO-101 and the ClothesNet meshes, is **standalone Newton with SolverVBD**, using `cloth_franka` as the template.

**Why Newton.**

- It runs on any Maxwell-or-newer NVIDIA GPU with driver 545+, on Linux or Windows ([Newton README](https://github.com/newton-physics/newton)).
- It imports the SO-101 from URDF or USD.
- It takes triangle meshes directly.
- It ships the only maintained example of an arm folding a T-shirt by friction, with robot μ 1.5, self-contact μ 0.25 and 10 substeps.
- It is where NVIDIA is putting its cloth work. Newton 1.0 went GA at GTC 2026 with VBD "two-way coupling with rigid-body solvers like MuJoCo Warp" ([NVIDIA blog](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)), and Isaac Lab 3.0 is building its deformables around VBD ([IsaacLab discussion #6249](https://github.com/isaac-sim/IsaacLab/discussions/6249)).

**Speed, single stream.** About 30 FPS for one scene on an RTX 4090, so a 900-frame episode takes roughly 30 s, about 30 times faster than IPC.

**Speed, parallel.** Do not expect parallel scaling. VBD cloth throughput in Isaac Lab "does not scale with environment count", plateauing around 5,800 steps/s and reaching **0.42× PhysX** at 2,048 environments on an RTX 5090 ([newton #4141](https://github.com/newton-physics/newton/issues/4141)). Plan for a few environments per GPU and benchmark before any paid run.

**Make the first week a go/no-go test, not a port.**

1. Load the SO-101 URDF and one ClothesNet shirt rescaled to centimetres. Set mass from 0.187 kg/m² and bending from the cantilever measurement.
2. Script four micro-tests:
   - a corner pinch-and-lift that holds for 10 s, measuring slip in mm;
   - a 20 cm lateral carry;
   - a jaw opening, measuring the cloth displacement that follows the fingers;
   - a half-fold left for 10 s, measuring creep.
3. Log the gripper joint torque during the pinch. A non-zero, grip-dependent torque proves force reaches the robot.

The coupling direction must be measured rather than assumed. The September 2025 NVIDIA blog described `cloth_franka` as "one-way coupling, where the rigid body affects the deformable, but not the other way around" ([NVIDIA blog](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/)). The 1.0 announcement claims two-way coupling, and the API notes that `integrate_with_external_rigid_solver=True` is one-way ([SolverVBD API](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverVBD.html)).

### Referee: Genesis IPC on the same micro-tests

Run the same four micro-tests in **Genesis with the IPC coupler**. Use `two_way_coupling=True`, `set_ipc_link_filter()` on the jaw links, d̂ 1.5 mm, cloth thickness from the real fabric, and friction from the incline test. This costs about 15 to 20 minutes of GPU time per 900-frame episode on a mid-range card, and likely more with 10 to 15k vertices than in the 7k-vertex demo. That is affordable for a dozen tests and for a scripted-oracle fold ceiling, not for a training dataset.

Where Newton and IPC agree with each other and with a phone video of the real arm doing the same pinch, adopt Newton. Where they disagree, IPC's penetration-free answer is the better estimate of reality.

pyuipc also installs by pip on Windows if the team prefers to drive libuipc directly ([libuipc](https://github.com/spiriMirror/libuipc)), but then the gripper control and rendering are theirs to write.

**Only after Newton passes, port the pipeline to Isaac Lab 3.0 + Newton VBD on Isaac Sim 6.x.** That route keeps USD assets, Isaac rendering and the camera set-up a VLA needs. It has a ready `Isaac-Lift-Cloth-Franka` preset (`newton_mjwarp_vbd_proxy`) to copy, but it is beta.

### Options to skip

**PhysX surface deformables on 5.1.** Static friction is unsupported, a forum respondent says rigid contact is not fully supported, and every garment would need re-authoring.

**MuJoCo flex.** Grasp slip is unresolved and the MJWarp flex port is incomplete.

**C-IPC, ThinShellLab, DiffCloth, ArcSim, SOFA and Bullet as the main simulator.** They are too slow, have no robot coupling, or reproduce today's problems. ThinShellLab is worth a look only if crease set proves decisive.

**FLASH and RGBench.** Watch them. If code appears, FLASH is the strongest validated candidate.

### What leaving LeHome costs

**Kept.** Garment meshes carry over after cleaning to single-surface triangle meshes. The SO-101 carries over via URDF or USD.

**Lost.**

- LeHome's task definitions and success metric, which must be reimplemented.
- The organisers' evaluator, so results no longer compare to the leaderboard.
- The camera set-up and rendering, which standalone Newton does not provide in a form the notes document.
- The teacher policy `lehome_sim`. It was trained on PBD dynamics and LeHome images, so it will not transfer without retraining.

**Effort estimate.** A realistic budget is one to two weeks for the Newton micro-tests and an IPC referee. Add several more weeks to rebuild a task, cameras and a scripted oracle in Isaac Lab 3. That is an estimate, not a sourced figure.

### If the teacher must stay on LeHome

Fix only what is clearly a bug, then re-evaluate the teacher after each change, because it was fitted to the old dynamics.

**Mass.** Set `particle_mass` to about 3.5 × 10⁻⁶ kg, giving a total of about 50 g, and set `gravity_scale` back to 1. Expect cloth-robot dynamics to change: a light cloth gets pushed aside instead of acting as an immovable mass.

**Sticking on release.** Set `adhesion` to 0 and lower `particle_adhesion_scale`. Switch the jaw colliders to convex hull, which addresses penetration.

**Slip.** Raise particle friction stepwise from 0.5 toward 5 to 25, following DexGarmentLab's 25 ([Particle_Garment.py](https://github.com/wayrise/DexGarmentLab/blob/e4e298e696bae5d866ded3b31e0ae27becea5376/Env_Config/Garment/Particle_Garment.py)). Step the simulation at 240 to 360 Hz.

**Last resort for carries.** Use a GarmentLab-style attachment block that binds only the particles between closed jaws ([GarmentLab Control.py](https://github.com/GarmentLab/GarmentLab/blob/6ee0620fddb8083d5eab554fa343cdc966c61d5f/Env/env/Control.py)).

**Creases and creep.** None of these settings fixes them. That is the solver. Where scoring uses the organisers' unmodified environment, follow the winner's lead: make the policy robust to slip, since he "did not retune" physics and attributed failed grasps to the simulator ([arXiv 2606.27163v2](https://arxiv.org/html/2606.27163v2)).

## Unverified items, conflicts and gaps

**Unverified claims used above.**

- The Newton `cloth_franka` parameters (μ 1.5, self μ 0.25, 10 substeps, centimetre units) came through a page summariser.
- Genesis's licence (Apache-2.0) and native URDF import are unconfirmed, and the current IPC coupler page returned 404, so the v0.3.14 API may have moved.
- That libuipc or VBD releases cloth cleanly when the jaws open is inferred from the absence of an adhesion term, not tested.
- The KES jersey bending range is snippet-level. The Ejaz cantilever values are internally inconsistent, so only its formula and areal densities are used.
- The fabric-on-fabric, plastic and silicone friction ranges are interpolations.
- The doll shirt mass (40 to 55 g), edge length (4 to 5 mm) and IPC episode cost are my estimates.
- Whether C-IPC uses dihedral bending without plasticity is unverified.
- The MJX flex feature list is snippet-only.
- PhysX surface deformables reachable through `DeformablePrim` in 5.1 is snippet-only.

**Conflicts.**

- Newton coupling direction. The 2025 blog says one-way and the 2026 GA blog says two-way. This is unresolved, and the torque micro-test settles it.
- Surface deformable friction. The schema exposes a static-friction field, while the deformables documentation says "static friction is unsupported". This report follows the documentation.
- libuipc Python versions. The README says 3.10 to 3.14 and the website says 3.10 to 3.13. The README is newer.
- Genesis issue #199. It was closed as "fixed", but the fix was an attachment API, not physical cloth grasping.

**Gaps no source filled.**

- No real-robot validation of IPC or VBD cloth grasping.
- No benchmark that measures grasp slip, release or crease retention rather than shape.
- No per-frame timing for libuipc or VBD on a single 10 to 15k-vertex garment.
- No mapping from VBD or PhysX stiffness units to physical N/m or N·m.
- No VBD static-friction or plasticity model.
- No verified tabulated tensile modulus or KES values for 100% cotton single jersey.
- No measured friction of jersey on PLA, TPU or silicone gripper pads.
- No sim-to-real study at doll scale.
- No public SO-101 + IPC or SO-101 + Newton integration.
- No released code for FLASH, RGBench GarmentDynamics or IsaacIPC. IsaacIPC also lacks cloth and drives the robot kinematically, one way ([arXiv 2605.24339](https://arxiv.org/html/2605.24339)).
- No SolverStyle3DPro pricing or friction model.

## Conclusion

The field has quietly split in two. Benchmarks built for policy learning pin cloth to the gripper because no fast solver grasps it honestly. Simulators built for fidelity model friction with a real stick regime, but they are either slow (IPC), unreleased (FLASH, RGBench) or still experimental (Newton VBD). The team's six symptoms place it squarely in the gap between the two: a removed PBD solver asked to do friction grasping that even NVIDIA's newer stack demonstrates only with inflated coefficients. Changing simulators therefore buys mostly *contact* realism. Real material numbers buy shape realism, and the literature suggests shape matters less for slow folding than for dynamic motion.

The cheapest high-value contribution is to measure rather than migrate first. A cantilever strip, an inclined plane and four scripted pinch tests run in Newton, in IPC and on the real SO-101 would produce the first grasp-level sim-to-real comparison for a parallel-jaw arm on doll-scale jersey. That result decides whether a port is worth weeks of work, and the literature does not yet have it.
