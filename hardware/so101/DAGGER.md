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

## It starts off as the scripted fold

Without `--trust-deg` the network has to earn the fold.

- The first fold is the scripted fold, row for row. The network watches, and is trained afterwards.
- It drives a fold, inside 2 degrees, only when on the fold before its rows were within 2 degrees of the plan's on
  at least 90% of the rows. The run says which it is before anything moves.
- A fold where it falls under 90% hands the next fold back to the scripted fold.

`--trust-deg 0` keeps the scripted fold driving whatever the network has earned. `--trust-deg 2` lets the network
drive now.

## The intervention button

Press the Xbox button on the controller and the fold waits on the row it is at. Press it again and the fold goes on.
While it waits

- the left stick moves the left arm's fingertip forward, back, left and right, and the right stick the right arm's
- LB raises the left fingertip and LT lowers it. RB and RT do the same for the right arm

The fingertip moves at 2 cm a second at most, 1 cm a second up and down, and no further than 3 cm from the plan on
any axis. A move that the arm cannot reach, or that brings it too close to itself, is not taken.

When the fold goes on, the fingertip keeps that offset for the rest of the fold. A nudge made before the corner is
lifted also moves where it is placed. A nudge made during the carry moves the carry and the place alone.

The offset is saved in `runs/dagger/nudges.json` and the next fold starts with it. So a correction holds from the
next fold on without any training. The corrected rows are what the network is taught, and what its 2 degrees are
measured from. Rows saved by folds before the offset changed are put aside in a `before_nudge_<time>` folder.

`--forget-nudges` starts again from no offsets. `--no-nudge` leaves the controller alone.

Check the controller first, with no arms. This prints what each control is read as.

```
python hardware/so101/dagger_nudge.py
```

On Windows the Xbox button also opens the Game Bar. Switch that off under Settings, Gaming, Xbox Game Bar. The
button is read through a call of XInput that Microsoft does not list, and that reading has not been tried on the
rig laptop. The check above shows whether it works.

With offsets stored, the first fold is the scripted fold moved by those offsets.

## Pictures

The phone's pictures and depth are saved through the fold, 5 a second, as a panel fold saves them. The script
opens the phone, takes the look's picture from it and records into the fold's run folder beside `steps.jsonl`. At
the end it asks whether the cloth folded and writes `label.json`. `rig_dataset.py list` then shows the run.

No pictures are saved with `--image`, `--log` or `--dry-run`. Those run as `fold_camera.py` runs them.

## On the rig, in order

Stop `ui.py` first, as for any script. Use the arguments you give `fold_camera.py fold`.

```
# once, no arms needed, about 15 minutes. The planned folds the network is trained on beside the rig's rows
python hardware/so101/dagger_data.py --episodes 900 --seed 0 --out outputs/hardware/dagger/base.npz

# nothing moves. Prints the share of rows the network would drive for the cloth in view
python hardware/so101/dagger_run.py --dry-run --left arm2 --right arm1

# a fold. The first one is the scripted fold with the network watching. Later ones are the network's once earned
python hardware/so101/dagger_run.py --left arm2 --right arm1 --speed 0.7 --torque-limit 50
```

The first command is only needed for the training after each fold. Until that file exists the folds still run and
their rows and pictures are kept, and the network stays as it is. It can run in a second terminal while you fold.

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

- The weights in the repo were trained on 648 planned folds, then corrected over six rounds in the arm model. On
  72 placements kept out of training the network's row is within 2 degrees of the plan's on 93% of rows. The plan
  drives the rest. Training them took two and a half minutes on a laptop.
- Without the scripted fold beside it the network still drifts off the fold. It is not yet a policy that can run
  alone.
- With the earlier weights (85% on the same test), on a simulated arm that lags and sags, folding one placement
  again and again took the share from 88% to 97% in three folds. Moving the cloth every fold moved the share on
  placements not yet folded from 88% to 90% in twelve folds. Rows from the arms sharpen the placements you fold.
- Training after a fold is 600 steps, a few seconds.

## Limits

- The network learns to reproduce the scripted fold with your offsets. An offset is one shift for the whole pinch
  and one for the whole carry, in the arm's own frame, wherever the cloth lies.
- The planned folds the network is also trained on have no offsets, so after a nudge it can lag the corrected rows
  and leave more of the fold to them.
- It reads the cloth's place from the scripted fold's look, the joints from the encoders and the time into the
  fold. It does not see the picture.
- In a recorded run the row logged as sent is the network's where it drove. The scripted fold's row for every
  moment is in the `rows_<time>.npz` named by the run's `dagger.json`.
- The planned folds have the bases 27 to 33 cm apart and the cloth's far edge 29 to 41 cm ahead. A cloth far
  outside that is left to the plan more often.
- When the two arms' plans come out with different row counts, `replay.py` refuses the fold before anything moves,
  as it does for the scripted fold. One of 900 drawn placements did this.
