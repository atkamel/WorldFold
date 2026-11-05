# Measuring neat folds beyond six keypoints

A neat fold should be defined as "every cloth vertex lies close to where a sequence of ideal reflections across named fold lines would put it, in a flat stack of the expected number of layers, with a compact rectangular footprint". It should be measured as a gated composite: the existing LeHome keypoint check, multiplied by a continuous score built from four terms. Those terms are rigid-aligned mean vertex distance to the ideal folded target, mirror-partner distance across each fold line, a height cap, and footprint rectangularity plus area ratio. The LeHome checker tests only **6 vertices with 4-5 pairwise 3D distance inequalities and no height or interior term**, which is why `lehome_sim` passes while leaving shirts bunched; every other simulation benchmark surveyed uses all vertices or a full silhouette. The dense vertex terms exist only in simulation, while the footprint terms (area ratio, rectangularity, template IoU, Canny wrinkle fraction) are what one overhead camera can measure, so the sim metric should compute both sets and the team should check how well they agree before trusting the camera subset on hardware. Published precedent covers the vertex-distance, mirror-partner, rectangularity and area-ratio terms; the height cap, the sleeve-tuck fraction, the specific weights and all thresholds are the team's own derivation and need calibration against hand-labelled neat and bunched states. For improving the policy, the cheapest steps need no training: sweep CFG scale, noise temperature and best-of-N against the new metric, change the stopping condition, and rank episodes. DSRL-style noise steering is the only method with a verified result on real cloth folding with a pi0-class model (25/30 from 2/30 in 90 rollouts) and fits a budget of tens of dollars. A full RECAP-style rerun as the teacher's author did it (about 12,500 episodes and about 300k steps at batch 192 on an H200) does not fit the budget as published. No source validates any sim neatness metric against human judgement, and no source RL-fine-tunes a pi0-class model against a neatness reward, so the team is ahead of the published evidence on both.

## The literature measures folds five ways, and the sparse ones are cheatable

### LeHome constrains six vertices and nothing else

The LeHome success checker reads all cloth vertex positions, keeps six "check point" vertices, converts to centimetres, and scales per-garment thresholds by the garment's `init_scale`. For tops, all five of these must hold: `dist(p0,p4) <= T0`, `dist(p2,p3) <= T1`, `dist(p1,p5) <= T2`, `dist(p0,p1) >= T3`, `dist(p4,p5) >= T4`. Long pants use `dist(p0,p4) <= T0`, `dist(p0,p2) >= T1`, `dist(p1,p3) >= T2`, `dist(p1,p5) <= T3`; shorts use `dist(p0,p1) <= T0`, `dist(p4,p5) <= T1`, `dist(p0,p4) >= T2`, `dist(p1,p5) >= T3`. Distances are full 3D norms, the check runs every 50th call, and the fold checker has no height term, although the fling checker in the same file does (`z_distance < 2` cm) ([success_checker_chanllege.py](https://github.com/lehome-official/lehome-challenge/blob/main/source/lehome/lehome/utils/success_checker_chanllege.py)). The winner's paper confirms the design: "There is no partial credit -- a fold passes all conditions or it fails", and "Binary success treats a barely-passing fold and a tight fold identically" ([Larchenko, arXiv 2606.27163](https://arxiv.org/abs/2606.27163)).

The consequence follows from the code. A legacy hard-coded index in the file reaches 8738, so the meshes have at least about 8,700 vertices, of which six are constrained. A sleeve keypoint sitting on a 5 cm bunch still satisfies a 3D threshold of the order of 10 cm, and the "stay apart" conditions only block balling the garment up along one or two axes. No published source documents a crumpled garment passing this check; that claim rests on the code and on the team's own observation.

### Formulas used in simulation

| Family | Formula | Source | Can a crumpled garment pass? |
|---|---|---|---|
| Keypoint-pair thresholds | LeHome conditions above; ClothesNet uses negative distance on 20 pre-selected vertices | [LeHome checker](https://github.com/lehome-official/lehome-challenge/blob/main/source/lehome/lehome/utils/success_checker_chanllege.py); [ClothesNet](https://arxiv.org/abs/2308.09987) | Yes |
| Tightness margin on the same check | worst per-condition margin, `1 - d` for proximity and `d - 1` for spread, with `d = dist/threshold` | [Larchenko](https://arxiv.org/abs/2606.27163) | Yes; it sees the same six points |
| Mirror-partner distance | `mean(‖pos_a − pos_b‖) + 1.2 · mean(‖pos_b − pos_b_init‖)`, reward is the negative; normalised as `(perf − perf_init)/(0 − perf_init)` | [SoftGym cloth_fold.py](https://github.com/Xingyu-Lin/softgym/blob/master/softgym/envs/cloth_fold.py) | Unlikely; bunching separates partners |
| Mean vertex distance to a goal | mean over all vertices of `‖v − g‖`; pose-invariant form `R_C = −‖v − g'‖` after a rigid planar fit, `R_A = −‖g' − g‖`, `R_CA = (1 − α)R_C + αR_A` | [FabricFlowNet](https://arxiv.org/abs/2111.05623); [Cloth Funnels](https://arxiv.org/abs/2210.09347) | No, if the threshold is tight |
| Area-weighted distance to a target folded mesh | `D = (1/|S_t|) Σ_i ‖q_i − y_i‖ A_i` over triangle barycentres | [Li et al. 2015](http://www.cs.columbia.edu/~yli/papers/YLI_IROS2015_Final.pdf) | No |
| Exponential goal reward | `r = exp(−D(s', g))` | [DaXBench](https://arxiv.org/abs/2210.13066) | No |
| Half-overlap | 2D IoU between the projected two halves of the cloth | [AdaFold](https://arxiv.org/abs/2403.06210) | Partly |
| Coverage | mask pixels divided by flattened-state pixels; `NI = (s − s0)/(smax − s0)` | [FlingBot](https://arxiv.org/abs/2105.03655); [VCD](https://arxiv.org/abs/2105.10389) | Yes; documented |
| Footprint shape | rectangularity = garment area / bounding-rectangle area; area ratio = final area / initial area; success = rectangularity above one threshold and area ratio below another | [MetaFold](https://arxiv.org/abs/2503.08372) | Possibly; a compact blob can be fairly rectangular and small |
| Point-cloud containment | final point cloud inside a box built from four manipulation points, `threshold=0.12` | [DexGarmentLab](https://github.com/wayrise/DexGarmentLab/blob/main/Env_StandAlone/Fold_Tops_Env.py) | Possibly |
| Hold condition | success must be maintained "for at least five seconds" | [GarmentLab](https://arxiv.org/abs/2411.01200) | Independent; rejects spring-back |

The "can a crumpled garment pass" column is reasoning from the formulas, not a cited result, except for coverage. SpeedFolding shows two shirts at 92.3% and 93.0% coverage that a learned classifier scored as smoothed and not smoothed, and concludes coverage "is insensitive to wrinkles" ([SpeedFolding](https://arxiv.org/pdf/2208.10552)). Cloth Funnels found downstream folding success of **87.8% from canonicalised states against 19.6% from coverage-optimised states** ([Cloth Funnels](https://arxiv.org/abs/2210.09347)). GarmentLab abandoned coverage for vertex distance as garments grew more complex ([GarmentLab](https://arxiv.org/abs/2411.01200)).

The remedy is the same each time: replace a sparse or silhouette proxy with a dense, correspondence-based distance to a clean reference, and add an independent term for whatever the first one misses. SoftGym's 1.2-weighted displacement penalty exists because a pure pairwise distance lets the agent "drag the cloth too far away while folding it" ([SoftGym](https://arxiv.org/abs/2011.07215)). Cloth Funnels factorised shape from pose because unaligned distance over-weights translation, and it takes the better of the goal and its mirror flip ([Cloth Funnels](https://arxiv.org/abs/2210.09347)).

No surveyed simulation benchmark reports height, curvature, normal variance, bending energy or layer count as a fold metric. Thresholds are not standardised either: Cloth Funnels chose its boundary "qualitatively", GarmentLab uses "a predefined value", and MetaFold does not print its thresholds.

### What the real world measures with a camera or a person

Real-world evaluation is still mostly human. SpeedFolding uses a majority vote of three reviewers ([SpeedFolding](https://arxiv.org/abs/2208.10552)). FoldNet++ uses human experts against a reference photo ([FoldNet++](https://arxiv.org/html/2609.12433)). pi*0.6 has raters judge "multiple quality metrics" that are aggregated into a success label ([pi*0.6](https://arxiv.org/html/2511.14759v2)). The LeHome real-robot final was jury-scored with partial credit, for example for a single folded sleeve ([Larchenko](https://arxiv.org/html/2606.27163)). A policy tuned to the six-point check is therefore exposed to an evaluation gap as well as a physical one.

Two rubrics are directly copyable. pi0's towel rubric gives one point per half-fold and one for "neatness of the final product" ([pi0](https://arxiv.org/html/2410.24164v1)). RoboFolDeX scores neatness 0-5 on "overall shape, compactness, flatness, structural stability, and edge alignment", requires two evaluators to agree, and combines it as `FoldScore = 35·Q̄ + 35·R + 30·max(0, 1 − t/T_max)` ([RoboFolDeX](https://arxiv.org/html/2609.10243)).

Five camera-computable measures have published definitions:

| Measure | Definition | Reference values | Source |
|---|---|---|---|
| Area ratio per fold | top-view area after / before each fold; target 1/2 per fold, 1/8 after three | error is 100% when nothing is folded or when final area falls below half the initial | [Garcia-Camacho benchmark sheet](https://ral-si.github.io/cloth-benchmark/resources/Protocols_and_benchmarks/Task2/Benchmark_Task2_RAL-SI-2020-B19-0832_2-V1.0.pdf); [Household Cloth Object Set](https://arxiv.org/abs/2111.01527) |
| Rectangularity | mask area / bounding-rectangle area | 0.85-0.94 on successful real folds | [MetaFold](https://arxiv.org/html/2503.08372) |
| Area ratio vs flat | final area / initial area | 0.24-0.45 on real folds | [MetaFold](https://arxiv.org/html/2503.08372) |
| Template IoU | IoU against the goal template "after rotating and translating the goal to best match" | 0.98 hand-placed, 0.74 human teleop, about 0.69 best autonomous | [Hoque et al.](https://arxiv.org/html/2204.10297) |
| Wrinkle penalty | fraction of interior mask pixels flagged by a Canny edge detector | 0.093 hand-placed, 0.088 human teleop | [Hoque et al.](https://arxiv.org/html/2204.10297); [Foldsformer](https://arxiv.org/abs/2301.03003) |

Each has a blind spot. Layers misaligned under the top layer do not change an overhead mask except by enlarging the footprint, which is why area ratio and IoU belong together. Lee et al. judged folds visually because IoU-type metrics are unreliable under self-occlusion ([Lee et al.](https://ar5iv.labs.arxiv.org/html/2010.03209)). The Canny fraction separated a perfect fold from human teleoperation by only 0.005, so it is meaningful only relative to a same-garment, same-camera, hand-folded baseline.

VLM judges are not yet a substitute. RL-VLM-F found Gemini-Pro "to perform poorly on Fold Cloth" and switched to GPT-4V, in simulation only ([RL-VLM-F](https://ar5iv.labs.arxiv.org/html/2402.03681)). No paper reports agreement between a VLM judge and human raters on real fold neatness.

### Reconciling the notes on rectangularity

The fold-procedures note concluded that rectangularity and aspect-ratio targets had no published precedent and were the team's derivation. The simulation-metrics and real-world notes both found that **MetaFold defines and uses rectangularity and area ratio**, in Isaac Sim and on a real xArm6 ([MetaFold](https://arxiv.org/html/2503.08372)). MetaFold wins: rectangularity has published precedent in both settings.

Three things remain the team's choice. MetaFold says "bounding rectangle" without stating axis-aligned or minimum-area rotated, so using the minimum-area rotated rectangle is a team decision. Its success thresholds are undisclosed, so a 0.85 pass mark is borrowed from the bottom of its reported real range, not from a stated rule. An explicit aspect-ratio target remains without precedent.

The notes also disagree on the simulated values. One reads rectangularity "around 0.8" and area ratio 0.33-0.6 from the PDF; the other reports 0.85-0.94 and 0.24-0.45 for the real robot. These are different experiments and both stand.

## Every garment reduces to a rectangle, and the target is computable

### Fold lines, mirror matrices and layer heights

The geometric line of work defines a garment as a planar polygon, a fold as a directed line segment whose endpoints lie on its boundary, and success as the part left of the segment lying flat on top of the rest. Each facet carries an integer stack height and a transformation matrix that is a product of the mirror matrices of the folds applied to it; after a fold, the folded facets' matrices are pre-multiplied by the new mirror and their height order is reversed and placed on top ([van den Berg et al. 2010](http://goldberg.berkeley.edu/pubs/WAFR-2010-Clothfolding.pdf)). That paper's Figure 1 is captioned "Folding a long-sleeve into a square using a sequence of seven g-folds", so the rectangle endpoint is literal.

Later systems kept the structure. Doumanoglou et al. derive folded polygon models "automatically from the original model of the spread garment" ([Doumanoglou et al. 2016](https://labicvl.github.io/docs/pubs/Andreas_TR_2016.pdf)). SpeedFolding's instruction is "a binary mask called template" plus "a list of folding lines relative to the template", with place poses "calculated by mirroring the pick poses at the folding line" ([SpeedFolding](https://arxiv.org/pdf/2208.10552)).

The sequence the team has in mind is the standard one. Doumanoglou uses two folds for towels, shorts and trousers and three for shirts: two side folds including sleeves, then one horizontal halving. Li et al. use the same counts ([Li et al. 2015](http://www.cs.columbia.edu/~yli/papers/YLI_IROS2015_Final.pdf)). UniFolding folds both sleeves along the fold line and then folds the garment in half ([UniFolding](https://arxiv.org/html/2311.01267)). BiFold's data has "the first two actions fold the sleeves, and the last action folds the trunk" ([BiFold](https://arxiv.org/html/2501.16458)). Towels are halved then halved again; pants go leg over leg, then in half across the length.

The endpoint is a convention. No source gives a normative aspect ratio, and BiFold notes that different step orders reach the same output. Consumer practice points the same way: KonMari folds sides to centre, then halves or thirds, and deliberately leaves "a little gap at the edge" ([KonMari](https://konmari.com/how-to-fold-clothes/)). An edge-alignment tolerance of a couple of centimetres is therefore correct rather than lenient.

### The sleeve fold line is a real choice

Two variants exist and give different target polygons. Variant (a) folds only the sleeve across the armpit-shoulder line, as in the g-fold interface where "the user first clicks on the left arm-pit, then on the left shoulder", and leaves a torso-width rectangle ([van den Berg et al. 2010](http://goldberg.berkeley.edu/pubs/WAFR-2010-Clothfolding.pdf)). Variant (b) folds along a vertical line inboard of the armpit, taking a strip of torso with the sleeve and giving a narrower rectangle. Doumanoglou's Figure 11 and FoldNet++'s "fold the half toward the midline" procedure are of this kind ([FoldNet++](https://arxiv.org/html/2609.12433)).

Scoring against the wrong variant penalises a good fold. The recommendation is to encode both as fold specifications, score against each, and take the better. This mirrors Cloth Funnels' mirror-flip rule and GPT-Fabric's minimum over symmetric final configurations ([GPT-Fabric repo](https://github.com/slurm-lab-usc/GPT-fabric-folding)). The team should then fix the primary variant after watching what `lehome_sim` actually does. Variant (a) is the likelier match, because the LeHome conditions require two keypoint pairs to stay apart, which a torso-width result satisfies; this is an inference, since the note-takers did not read which body locations the six keypoints mark.

Fold lines should be defined from armpit and hem keypoints, not shoulders. Doumanoglou localised shoulders worst (2-4 cm) and noted they "are not used for planning of the next folding move".

For short sleeves, variant (a) barely changes the silhouette. Sleeve tuck must therefore be measured on sleeve vertices, never inferred from area.

### Computing the ideal target per garment

The recipe is the team's synthesis of the g-fold data structure and published scoring practice. Take the flat rest-pose vertex array. Write each fold as a directed segment between two named keypoints, or fractions between them, with a flag for whether it applies to the whole stack or only the previously folded part. Apply the 2D reflections in order to the vertices on the folding side; this gives every vertex an ideal xy position and an integer layer index. Score the achieved state after a rigid SE(2) alignment with outlier filtering, taking the minimum over the left-right mirror flip and over the sleeve variants.

The precedent for an analytic reflected target is partial, and the notes differ here. One note found a search snippet describing a target "with the part to be folded geometrically rotated by 180 degrees" but could not pin the source paper. The other found no paper that states analytic reflection as its goal-generation method. What is verified is that Li et al. score against a target folded mesh, and that sim papers obtain goals by running a scripted fold from a perfect start (FoldNet, Foldsformer) or by saving a human demonstration's final state (GarmentLab). The analytic target is defensible but should be described as the team's construction.

The scripted or teleoperated clean fold is the fallback, and it has one advantage: it includes real bend radii. The pure reflection assumes zero thickness and infinite friction, so it is slightly too compact near creases. A tolerance of roughly fabric thickness times layer count near fold lines is expected.

### Flattening is a prerequisite, not a detail

Every pipeline reviewed assumes or first produces a flat, canonical garment. Cloth Funnels' folding heuristic "completely fails" from random configurations. Doumanoglou found 0 of 30 configurations had all landmarks undeformed before spreading and 25 of 30 after. Li et al. call flat placement "a prerequisite for garment folding". LeHome sidesteps this by starting garments "lying flat on a table" ([Larchenko](https://arxiv.org/html/2606.27163v2)).

Two consequences follow for the metric. First, record a start-state quality number, using Cloth Funnels' shape-versus-pose split after Procrustes alignment to the rest pose, so a bad start is not blamed on the fold. Second, score per-fold intermediate targets as well as the final state, since drift accumulating across folds is the dominant multi-fold failure.

## A gated five-term composite is the metric to implement first

The recommended metric is `Neat = G · (w_v·s_v + w_m·s_m + w_h·s_h + w_f·s_f)`, with each `s = clip(1 − e/τ, 0, 1)` and every length normalised by the geometric mean of the garment's flat height and width, as Cloth Funnels does. Starting weights of 0.4, 0.2, 0.2, 0.2 and all tolerances τ are placeholders with no published basis. Report every component alongside the composite, as SoftGym does with normalised performance.

| Component | Computation | Guards against | Status |
|---|---|---|---|
| Gate `G` | LeHome six-keypoint check true, held for several seconds | wrong fold entirely; spring-back; rewarding half-folds | Published: LeHome checker; hold from [GarmentLab](https://arxiv.org/abs/2411.01200) |
| Vertex distance `e_v` | mean planar distance of all vertices to the ideal target after SE(2) alignment, min over mirror flip and sleeve variant; also report the 95th percentile | bunching, shifted layers, wrinkles, misaligned edges at once | Mean and alignment published ([Cloth Funnels](https://arxiv.org/abs/2210.09347), [Li et al.](http://www.cs.columbia.edu/~yli/papers/YLI_IROS2015_Final.pdf)); percentile and analytic target are the team's |
| Mirror-partner distance `e_m` | per fold line, mean distance between each vertex and its partner across the line, from an index map built once on the rest mesh | interior separation that keypoints miss; needs no stored goal | Published for one fold on rectangles ([SoftGym](https://github.com/Xingyu-Lin/softgym/blob/master/softgym/envs/cloth_fold.py)); extension to sleeve and body lines is the team's |
| Height `e_h` | 95th-percentile vertex height above the table, minus expected layers times fabric thickness | the bunch itself: a neat stack is a few cloth thicknesses, a bunch is centimetres | Team derivation; only precedent is the 2 cm keypoint z-check in LeHome's fling checker |
| Footprint `e_f` | rasterise vertex xy to a mask; rectangularity against the min-area rectangle; two-sided error of area ratio against the target's expected ratio | sprawl and compact blobs; this is the term a camera can also compute | Rectangularity and area ratio published ([MetaFold](https://arxiv.org/abs/2503.08372)); two-sided target error follows the [Garcia-Camacho](https://ral-si.github.io/cloth-benchmark/) boundary behaviour; min-area rectangle is the team's choice |
| Sleeve tuck (tops, diagnostic) | fraction of sleeve vertices whose xy lies inside the target torso rectangle and whose layer is above the torso | short sleeves left out, which area cannot see | Team derivation; no paper names such a measure |
| Drift (diagnostic) | displacement of the stationary part from its initial position | dragging the garment while folding | Published ([SoftGym](https://arxiv.org/abs/2011.07215), 1.2 weight) |

Multiplying by the gate follows two independent precedents. pi*0.6 and Larchenko both keep the episode return tied to the outcome and use dense signals only for credit assignment; Larchenko withdraws all intermediate reward on failure because checkpoint conditions "may hold while the rest of the garment is in a state that makes success unlikely". A neatness reward without an outcome gate pays for half-folds.

Area ratio must be two-sided. Bunching shrinks the footprint as well, so "area ratio below a threshold" alone is satisfied by a crumple. The Garcia-Camacho quality function already assigns 100% error when final area drops below half the initial area for a single fold.

Everything here is a few NumPy operations on the vertex array the checker already fetches through `particle_object.get_current_mesh_points()`. Whether that call returns simulation-mesh or render-mesh vertices, and whether face topology is exposed, needs checking in the team's checkout.

### Carrying the metric to one overhead camera

On hardware, vertices are gone. What remains is the footprint term (rectangularity, area ratio against a flat-state reference photo), template IoU against a rectangle of the intended folded dimensions after rigid alignment, and the Canny wrinkle fraction relative to a hand-folded reference of the same garment under the same camera. Plain single-colour cloth on a contrasting table keeps segmentation tractable; the Garcia-Camacho protocol lists table and towel colour as declared assumptions for this reason.

The bridge between the two is the step no paper has taken. Render or rasterise the same camera metrics in simulation for every episode and measure how well they rank episodes compared with the full vertex composite. If the camera subset tracks the composite in sim, it is a credible real-world proxy. If it does not, the team needs keypoint markers or a human rubric on hardware. Either way, score a sample of real folds with two raters on the RoboFolDeX 0-5 scale and report agreement.

Trial counts matter for claims. With 15 trials, FoldNet++'s interval `sqrt(p(1−p)/(k−1))` is about ±13 points at p = 0.5, so differences of 10-20 points at typical sample sizes are noise.

## Improving the policy: a cost ladder from sweeps to RECAP

The cost anchor is the team's own measurement, recorded in project notes, of about $1.85 per 100 episodes with four workers. The research notes assumed $0.02-0.10 per episode. No paper read reports GPU-hours or dollar cost, so every budget judgement below is the team's inference from published episode counts and hardware.

| Rung | What is trained | Published scale | Fits tens to low hundreds of dollars? |
|---|---|---|---|
| 1. Sweep CFG, noise temperature, N, execution length | nothing | Larchenko tuned these with bandits during rollouts | Yes; a few hundred episodes |
| 2. Change termination; filter and rank episodes | nothing | no published evidence for termination | Yes |
| 3. Best-of-N with a neatness scorer | small head or none | N = 2-3 helped; scorer correlation "effectively zero" | Yes |
| 4. DSRL noise steering | small MLP actor and critics | 65-90 real episodes or about 10k sim samples | Yes |
| 5. Residual policy (PLD, ResFiT) | 3-layer MLP | 250k env steps in sim | Borderline |
| 6. Reward-weighted imitation (SARM RA-BC) | LoRA on pi0 | 2x RTX 4090, 40k steps, batch 32 | Plausibly low hundreds; cost not reported |
| 7. Action-expert PPO (pi_RL) | 300M action expert | 8x H100 | No |
| 8. Full RECAP-style rerun | whole VLA except vision encoder | 12,500 episodes, about 300k steps, batch 192, H200 | No as published; a reduced version is untested |

### Rungs one to three need no gradient through the model

The released checkpoint already contains an advantage token and uses classifier-free guidance, `v = v_uncond + α(v_cond − v_uncond)`, with final scales of 7 for both tops and short pants and 9 for long pants. Noise temperature was 0.7-0.9, execution length 3-5 actions and N = 2 or 3 ([Larchenko](https://arxiv.org/html/2606.27163v2)). These were tuned against binary success with the update rule `r = success − running success rate`. Re-running the same sweep with the neatness score as the bandit reward is the cheapest experiment available. It also probes how much neatness the existing "good action" direction contains, since training gave the top 20% tightest successes a 0.3 advantage bonus.

Changing the stopping condition is the team's own proposal and has no published evidence. The policy learned to stop where the threshold checker fires. Letting it continue until neatness plateaus tests whether it can tidy at all.

Best-of-N needs care. Larchenko reports the correlation between his scoring head and outcome "was effectively zero", although two or three candidates still beat one. A neatness head trained on chunk-level outcomes may fail the same way. In simulation there is a stronger, untested option: snapshot the state, roll each candidate forward briefly, and score with the true metric, at N times the sim cost per decision.

### DSRL is the best-evidenced trained step for cloth

DSRL trains a small MLP over the flow policy's input noise using only forward passes of the frozen base. It took a real Franka task from 2/10 to 9/10 in about 40 episodes, and steered real pi0 from 5/20 to 18/20 in 80 episodes ([DSRL](https://arxiv.org/html/2506.15799)). On real bimanual clothes folding with a frozen pi0 and 90 rollouts, a dense progress reward reached **25/30, against 7/30 with sparse reward and 2/30 for the base** ([RARM](https://arxiv.org/html/2606.22027), Table 1; corrected 2026-11-05, this report first said 12/30, which is RARM at 60 rollouts). That dense-versus-sparse gap is the strongest published argument that a continuous neatness reward helps a noise-steering policy, though that test had no reset bank or SGS sampling as far as we read.

Two cautions apply. The noise actor acts on the same variable that best-of-N samples and that temperature scales, so it replaces those features and does not stack with them; no paper tests DSRL on a policy that already uses CFG and best-of-N. And the published gains start from weak bases. The teacher already succeeds 70-93% of the time by category, so the question is whether steering can move neatness at fixed success, which nobody has measured.

GR-RL shows the same idea at larger scale: a 51.5M-parameter noise predictor on a frozen 5B VLA lifted shoe-lacing from 72.7% to 83.3% ([GR-RL](https://arxiv.org/html/2512.01801)). PLD's residual MLP reached 99.2% on LIBERO-90 with OpenVLA but used 250k environment steps ([PLD](https://arxiv.org/html/2511.00091)). At about 30 seconds per episode that many steps is the borderline case for this budget.

### Reward-weighted imitation is the cheapest route that changes the weights

SARM's RA-BC weights each training chunk by predicted progress gain: `w̃ = clip((r̂ − (μ − 2σ))/(4σ + ε), 0, 1)`, with weight 1 above κ = 0.01 and 0 for negative progress. Fine-tuning pi0 with LoRA on two RTX 4090s for 40k steps at batch 32 took T-shirt folding from a flattened start from **8% to 83% (10/12)**, and from crumpled from 0% to 67% ([SARM](https://arxiv.org/html/2509.25358)). GR-RL gained 15.9 points just by dropping transitions where its learned progress function fell.

In simulation the team does not need SARM's learned reward model, because the vertex metric is exact and free. It needs only the weighting rule: weight chunks by the gain in the neatness score. Learned progress models (SARM, RARM, GVL) become relevant only for scoring real-robot footage.

One caveat is that these results measure success, not neatness. The only published neatness-directed weighting is Larchenko's precision boost, which he did not ablate.

### Full RECAP as the teacher's author did it

RECAP trains the VLA with a binary improvement indicator and the objective `−log π(a|o,l) − α·log π(a|I_t,o,l)`, drops the indicator randomly, and applies CFG at inference. Its reward is −1 per step, 0 on success and −C_fail on failure, with a 670M-parameter distributional value function ([pi*0.6](https://arxiv.org/html/2511.14759)). Its abstract claims it "more than doubles task throughput and roughly halves the task failure rate" on the hardest tasks.

Larchenko combined AWR, as sampling proportional to `exp(clip(A, −2, 2))`, with a RECAP-style advantage token and GAE over learned success and completion heads. He froze only the SigLIP encoder and ran about 12,500 rollout episodes (about 4.3M frames) and about 300k steps at batch 192 on a single H200, with a fresh checkpoint about every 40 minutes ([Larchenko](https://arxiv.org/html/2606.27163v2)). His own caveat is "This is an engineering case study, not a controlled experiment".

Reward enters his pipeline in three places: per-frame reward and checkpoint logic at rollout time, advantage recomputation before each training iteration, and the precision-boost ranking. The smallest change is the third: replace the six-point tightness margin with the vertex composite and keep the binary return. Existing rollouts can be relabelled only if they stored cloth vertices, which is unconfirmed. A reduced rerun of a few hundred to two thousand fresh episodes and a few thousand steps from the released checkpoint is the only version near budget, and it is untested. Fitting on a 48 GB L40S would need a smaller batch or LoRA, which the repository is not confirmed to support.

The recommended order is rung 1, then 2, then 4, with rung 6 as the first step that touches weights. Rung 8 should wait until a cheaper rung shows the metric is learnable.

## Rigid SO-101 jaws are the least favourable documented grasp setup

Published systems almost never pinch flat cloth off a hard table with plain rigid jaws. Van den Berg used "a soft working surface, so the relatively thick grippers can easily get underneath the cloth". SpeedFolding added 3D-printed fingertip teeth and a thin sponge mattress, and approaches by hovering 4 cm above, rotating 8 degrees so one fingertip is lower, then moving 1 cm toward the higher one; it reports grasp success above 96% on known garments ([SpeedFolding](https://arxiv.org/pdf/2208.10552)). Lips et al. printed 8 cm-wide TPU fingertips "to ensure compliance of the fingers while sliding underneath the cloth", and light towels were still pushed away for lack of friction ([Lips et al. 2022](https://arxiv.org/pdf/2205.06714)). Doumanoglou's lower finger slides under the garment because the jaw "is not suitable for grasping flat garment from above".

Grasp type affects fold quality directly. Borras et al. found fold quality "clearly worst" for the pinch grasp, though they did not quantify it, and argue a fold improves "if a folding line were rigidly maintained" ([Borras et al. 2020](https://arxiv.org/pdf/1906.08202)). Lips attributes about **80% of fold failures to corner misalignment** as corners bend during the motion, and suggests bimanual manipulation or wider fingertips. The team's bimanual setup, holding both ends of the moving edge, is the configuration the g-fold model requires.

Layer selection is the other limit. UniFolding states that where the grasp point has several layers the gripper cannot take just one ([UniFolding](https://arxiv.org/html/2311.01267)). Van den Berg's failures concentrated on re-grasping a previously folded sleeve, where the robot either grasped nothing or "heavily bunched the cloth". That is the bunching the team sees, and it argues for logging which fold step introduces the neatness loss.

For the SO-101 itself the evidence is thin. A second LeHome participant reports "many failures came from grasp quality: missed cloth pickup, weak grasp closure, or unstable contact" ([Aarvis write-up](https://aarvis.github.io/lehome2026/)). Larchenko lists physics simplifications that cause "a garment slipping out of the gripper" as a failure class. Community TPU fingertips and friction tape exist, but no peer-reviewed measurement of SO-100 or SO-101 jaws on fabric was found.

The practical implication is to log gripper-cloth contact loss separately from final-shape error. Otherwise simulator slip and hardware grasp failure are indistinguishable from policy sloppiness in the neatness number. The supported hardware mitigations are a thin compliant mat, compliant high-friction fingertip pads, and a tilted approach.

## What remains unverified

Nothing in this report was re-checked against PDFs during synthesis. The notes were produced largely through a fetch-and-summarise tool, and the items below need a second look before they carry a decision or a citation.

| Item | Status | Action |
|---|---|---|
| Crumpled garments passing the LeHome check | Team observation plus code reading; no published report; GitHub issues and Discord not checked | Quantify on the team's own rollouts |
| LeHome `check_points` indices and `success_distance` values | Not read; they live in garment config files | Read locally; confirm which body locations p0-p5 mark |
| `get_current_mesh_points()` mesh type and topology access | Not verified | Check `Garment.py` |
| Garcia-Camacho quality function algebra | Only boundary behaviour confirmed; paper PDF returned 404 | Open the spreadsheet |
| MetaFold thresholds and rectangle type | Not disclosed in text read | Calibrate own thresholds |
| FoldNet "average vertex error below 0.4 mm" | Summary only; implausibly tight | Re-read PDF |
| BiFold 0.0125 m and IoU > 80%; UniFolding, UniGarmentManip, DexGarmentLab details | Summarised HTML only; PDFs exceeded fetch limit | Spot-check |
| DexGarmentLab `judge_pcd` meaning of 0.12 | Function body not located | Read repository |
| Cloth Funnels α = 0.6 and τ = 0.3 | Greek symbols stripped in extraction and restored from context | Confirm in PDF |
| "Target rotated by 180 degrees" per-vertex metric | Snippet; source paper not pinned (candidates: Sarker et al. 2021, Lips et al. 2024) | Find the source or drop the citation |
| PR2 results 9/9, 7/9, 7/9, 5/9 | One note attributes them to van den Berg 2010 (read), another to Miller 2012 (snippet) | Settle the attribution |
| Miller et al. 2012 full text; Stria et al. 2014; Petrik et al. | Not opened; snippets only | Read before citing fold-line definitions |
| Hoque et al. IoU and wrinkle numbers | Fetched in one note, snippet in the other; a search summary wrongly attributed them to SpeedFolding | Confirm the table |
| Cloth Funnels real-world 0.65 IoU and 0.80 coverage | Low confidence; tool summary was garbled | Re-read |
| RoboFolDeX rubric levels 1-4; FoldNet++ | Fetched summaries; intermediate levels not extracted | Read the appendix before copying the rubric |
| SoftGym paper wording | Search-result summary in one note; the code was read directly | The code is authoritative |
| pi*0.6 laundry throughput | Notes disagree: about 1.5 to 2.5 tasks/hour on simple laundry in one, about 4 to over 8 per hour on diverse laundry in the other (snippet) | Re-check both |
| RECAP advantage threshold "30% percentile" | Semantics unclear | Re-check |
| GVL, RL-VLM-F numbers, VLA-RL, PA-RL, ConRFT, HIL-SERL, RLDG, ReinFlow, DPPO, FQL, RIPT-VLA, SimpleVLA-RL | Abstract only | Do not rely on step counts or compute |
| ResFiT, Policy Decorator, Rank2Reward | Search summaries only | Unverified |
| GR-RL online episode count and filter threshold | Not extracted | — |
| All full-text numbers for RECAP, pi_RL, DSRL, PLD, SARM, RARM, Larchenko | Extracted from HTML by a summariser; not cross-checked against tables | Re-check any figure a decision hinges on |
| Where reward, checkpoints and advantages are computed in `lehome_solution`; CLI flags for CFG and best-of-N; LoRA support; whether stored rollouts contain vertices | Not located | Read the repository |
| Dollar costs | No paper reports cost or GPU-hours; all budget fits are inference | Measure a pilot |
| ALOHA 2 gripper claims; "rubber fingertip" claims; Clark et al. 2023; SO-101 TPU tips | Snippets or hobbyist listings | Treat as leads |
| Longhini et al. 2024; Sanchez et al. 2018; De Gusseme et al. 2025 | Summary, second-hand, and 403 respectively | — |
| Retail folding-board sizes; towel-fold guides | Vendor listings and consumer guides; no formal standard found | Not load-bearing |
| Wall-OSS and "Fold Towel" 0-3 staged rubrics | Snippet only; source unconfirmed | Excluded from recommendations |
| Company demos (Figure, 1X, Dyna, Generalist) | No rubric found; articles not opened | — |
| Height, normal, curvature and layer-count metrics; sleeve-tuck fraction; termination change; look-ahead best-of-N; composite weights | Team proposals with no published precedent or reference thresholds | Calibrate on hand-labelled states |
| Correlation of any sim neatness metric with human judgement | No source found | The team must measure this |

## Conclusion

The failure the team observed is the predictable result of optimising a strong policy against a six-point proxy, and the literature's repeated answer to proxy failure is a dense, correspondence-based distance plus one independent term per remaining blind spot. The weak link is not the formula but calibration. No published threshold exists for any neatness term, and no one has shown that a sim vertex metric tracks what a human or a jury calls neat. A small labelled set of neat and bunched final states, scored by the composite, by the camera subset and by two raters, would settle that question and would be new evidence.

The ladder also reorders the plan. Because the teacher already carries an advantage token trained partly on fold tightness, the first experiment costs a few dollars and tells the team whether neatness is reachable by steering alone. If steering works, the full RECAP rerun is unnecessary. If the policy cannot tidy even when allowed to continue, the bottleneck is likely grasp physics rather than reward, and money is better spent on fingertips and a mat than on GPUs.
