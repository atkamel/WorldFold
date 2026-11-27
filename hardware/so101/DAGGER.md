# A learned policy on the camera fold, corrected as it runs

`dagger_run.py` plays the camera fold with a small network proposing every row of joint targets. The scripted fold
stays in charge. It plans the fold from the picture as it always does, and its row replaces the network's whenever
the two differ by more than a set number of degrees. After each fold the network is trained on what it saw, with
the scripted fold's rows as the answers. This is DAgger with the scripted fold as the teacher.

No file of the scripted fold is changed. `dagger_run.py` calls `fold_camera.py fold` as it is and swaps two
functions in memory while it runs. `fold_traj.fold` is called as before and its inputs are kept. `replay.play`,
the loop that sends the rows, is replaced by one that asks the network first. The look, the saved pinch settings,
the checks, the P gain, the torque cap, the glides and the run log are the scripted fold's own.

Not yet run on the arms. Checked on the stand-in servo bus only (`test_dagger.py`).

## What reaches the arms

For every row and each arm

1. the arm's joints are read
2. the network proposes the row
3. its row is sent when every joint is within `--trust-deg` of the plan's row, the fingertip is no more than 3 mm
   under the plan's, and the pose is as clear of the arm itself as `replay.py` asks of a plan
4. otherwise the plan's row is sent

Either row goes through `replay.py`'s rate cap and is reached in the same equal moves. The last row is the plan's,
and a row or two are added if the rate cap left an arm short of it. So the arms stay inside a band of `--trust-deg`
around the scripted fold, and `--trust-deg 0` is the scripted fold with the network only watching.

The fingertip rule is there because 2 degrees at the joints can lower the fingertip by 15 mm at the pinch, and the
plan's pinch heights already sit at the table.

The scripted fold drives every row, and nothing is learned from the run, with `--grasp-check`, `--no-gripper` or a
`--closed` other than -0.17.

## On the rig, in order

Stop `ui.py` first, as for any script. Use the arguments you give `fold_camera.py fold`.

```
# once, no arms needed, about 15 minutes. The planned folds the network is trained on beside the rig's rows
python hardware/so101/dagger_data.py --episodes 900 --seed 0 --out outputs/hardware/dagger/base.npz

# nothing moves. Prints the share of rows the network would drive for the cloth in view
python hardware/so101/dagger_run.py --dry-run --left arm2 --right arm1

# the scripted fold drives, the network watches and is trained afterwards
python hardware/so101/dagger_run.py --trust-deg 0 --left arm2 --right arm1 --speed 0.7 --torque-limit 50

# the network drives inside 2 degrees of the scripted fold
python hardware/so101/dagger_run.py --trust-deg 2 --left arm2 --right arm1 --speed 0.7 --torque-limit 50
```

After each fold it prints, per arm, how many rows the network drove, why the plan drove the others, and how far
the two were apart. Then it trains for a few seconds and names the weights the next run starts from.

## What is saved

In `runs/dagger/` (not in git)

- `rows_<time>.npz` holds, per row and arm, what the network read, what it proposed, the plan's row, and which was
  sent
- `round_<n>.npz` and its `.json` hold the weights after each fold. The newest one is used by the next run.
  `--policy` picks another

The run folder of the fold has the usual `steps.jsonl`, with the rows as sent and a `dagger` line per row naming
who drove each arm. To go back to the weights in the repo, move the `round_*` files away.

## What to expect

Measured in the arm model only (results.md on `feat/cloth-world-model`, 2026-11-26).

- The weights in the repo were trained on 648 planned folds. On 72 placements kept out of training the network's
  row is within 2 degrees of the plan's on 85% of rows. The plan drives the rest.
- Without the scripted fold beside it the network drifts off the fold within a few rows. It is not yet a policy
  that can run alone.
- On a simulated arm that lags and sags, folding one placement again and again took the share from 88% to 97% in
  three folds. Moving the cloth every fold moved the share on placements not yet folded from 88% to 90% in twelve
  folds. Rows from the arms sharpen the placements you fold. More planned folds (`dagger_data.py --episodes`) are
  what widens it.

## Limits

- The network learns to reproduce the scripted fold. It will not fold better than the script does.
- It reads the cloth's place from the scripted fold's look, the joints from the encoders and the time into the
  fold. It does not see the picture.
- A fold from the command line saves no pictures, so these runs do not yet add to a camera dataset.
- The planned folds have the bases 27 to 33 cm apart and the cloth's far edge 29 to 41 cm ahead. A cloth far
  outside that is left to the plan more often.
- When the two arms' plans come out with different row counts, `replay.py` refuses the fold before anything moves,
  as it does for the scripted fold. One of 900 drawn placements did this.
