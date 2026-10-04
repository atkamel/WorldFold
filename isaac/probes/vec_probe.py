"""V0 go/no-go probe (throwaway): B independent copies (cloth 101x101 with its own particle system + material, arm
pair, table) in ONE Isaac process on the GPU weld profile. Copy 0 is the normal scene; copies 1..B-1 sit 1.5 m apart
along x. Measures s per control step (zero actions) and process VRAM, and checks each copy's cloth view reads/writes
independently. Usage: python vec_probe.py B"""
import json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
SPACING = 1.5


def vram_mb():
    # Windows/WDDM reports no per-process memory; the total in use on the GPU (all processes) instead
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


def main():
    B = int(sys.argv[1])
    from isaac.isaac_env import start_app
    start_app(True, False, device="cuda:0")
    import torch
    from omegaconf import OmegaConf
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from lehome.assets.object.Garment import GarmentObject
    from isaac import lab_scene as L
    from isaac.isaac_env import MUJOCO_ARM_DRIVE, TABLE_SIZE, IsaacClothFoldEnv
    from mujuco.cloth_params import ARM_BASE_LEFT, ARM_BASE_RIGHT, TABLE_TOP_Z

    class MultiScene(L.SceneEnv):
        def _setup_scene(self):
            super()._setup_scene()
            self.extra = []
            for i in range(1, B):
                dx = np.array([SPACING * i, 0.0, 0.0])
                arms = {}
                for prefix, base in (("left_", ARM_BASE_LEFT), ("right_", ARM_BASE_RIGHT)):
                    cfg = L._robot(f"/World/Copy{i}/{prefix}Robot", np.asarray(base) + dx, MUJOCO_ARM_DRIVE)
                    arms[prefix] = Articulation(cfg)
                    self.scene.articulations[f"copy{i}_{prefix}arm"] = arms[prefix]
                table = sim_utils.CuboidCfg(size=(TABLE_SIZE, TABLE_SIZE, TABLE_TOP_Z),
                                            collision_props=sim_utils.CollisionPropertiesCfg())
                table.func(f"/World/Copy{i}/table", table, translation=(dx[0], 0.0, TABLE_TOP_Z / 2.0))
                # the same cloth asset and particle config as copy 0, moved by dx (its own particle system/material)
                pcfg = OmegaConf.load(L.PARTICLE_CFG)
                n = (L.CLOTH_COUNT - 1) * L.CLOTH_SUBDIV + 1
                pcfg.objects.garment_config.particle_mass = L.CLOTH_MASS / (n * n)
                pcfg.objects.particle_material.gravity_scale = L.CLOTH_GRAVITY_SCALE
                pcfg.objects.particle_material.adhesion = L.CLOTH_ADHESION
                pose = [float(v) for v in (self.cloth_pose + dx)]
                gcfg = OmegaConf.create(dict(OmegaConf.to_container(self._garment_cfg_proto), **{
                    "initial_pos_range": pose + pose, "soft_reset_pos_range": pose + pose}))
                cloth = GarmentObject(f"/World/Copy{i}/cloth", pcfg, gcfg, rng=np.random.RandomState(i))
                self.extra.append({"arms": arms, "cloth": cloth})

        def _spawn_cloth(self):
            cloth, rest = super()._spawn_cloth()
            self._garment_cfg_proto = cloth.garment_config
            return cloth, rest

    L.SceneEnv = MultiScene
    env = IsaacClothFoldEnv(observation_mode="state", max_episode_steps=1000, profile="weld")
    lab = env.lab
    for e in lab.extra:
        e["cloth"].initialize()
    env.reset(seed=0)
    rep = {"B": B}
    # independence: each copy's cloth view reads its own particles at its own x offset
    views = [lab.cloth._cloth_prim_view] + [e["cloth"]._cloth_prim_view for e in lab.extra]
    xs = [float(v.get_world_positions()[0][:, 0].mean()) for v in views]
    rep["cloth_mean_x"] = [round(x, 3) for x in xs]
    rep["copies_separate"] = all(abs(xs[i] - xs[0] - SPACING * i) < 0.05 for i in range(B))
    # per-copy mass write: scale copy B-1's masses, check copy 0 unchanged
    if B > 1:
        pv0, pvk = views[0]._physics_view, views[-1]._physics_view
        m0 = pv0.get_masses().clone()
        mk = pvk.get_masses().clone()
        pvk.set_masses(mk * 2.0, torch.arange(mk.shape[0], device=mk.device))
        rep["mass_write_isolated"] = bool(torch.allclose(pv0.get_masses(), m0) and
                                          torch.allclose(pvk.get_masses(), mk * 2.0))
    a = np.zeros(14, dtype=np.float32)
    for _ in range(5):
        env.step(a)
    t0 = time.time()
    N = 40
    for _ in range(N):
        env.step(a)
    rep["s_per_step"] = round((time.time() - t0) / N, 4)
    rep["vram_mb"] = vram_mb()
    print("VEC", json.dumps(rep), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
