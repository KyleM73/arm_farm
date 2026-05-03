# tasks

Registered mjlab tasks for SO-ARM101. All share the same robot, action
space, physics rate (200 Hz), and policy rate (~28.6 Hz, decimation=7).

| Task         | Actor obs                                                       | Critic obs (privileged)            | Camera obs               | Reward terms | Events extra to base |
|--------------|-----------------------------------------------------------------|------------------------------------|--------------------------|--------------|----------------------|
| `Play`       | `joint_pos(6) + joint_vel(6) + actions(6)` = **(18,)**          | —                                  | —                        | —            | —                    |
| `Cube`       | `joint_pos(6) + joint_vel(6) + ee_to_cube(3) + cube_to_goal(3) + actions(6)` = **(24,)** | same as actor                      | —                        | 5 terms      | —                    |
| `Cube-Rgb`   | `joint_pos(6) + joint_vel(6) + actions(6) + goal_position(3)` = **(21,)** | `joint_pos + joint_vel + ee_to_cube + cube_to_goal + actions` = **(24,)** | `front`: `(3, 32, 32)` RGB float32 in `[0, 1]` | 5 terms | `cube_color` (RGBA DR per reset) |
| `Cube-Depth` | `joint_pos(6) + joint_vel(6) + actions(6) + goal_position(3)` = **(21,)** | same as `Cube-Rgb` critic = **(24,)** | `wrist`: `(1, 32, 32)` depth float32 clamped to `[0, 1]` via 0.5 m cutoff | 5 terms | —                    |

Action: `joint_pos` — relative joint targets, `(6,)`. Each output is
multiplied by `SO101_ACTION_SCALE` (≈ ±2.4° per step at ±1) and added to
`default_joint_pos`.

Reward terms (all five tasks except `Play`): `lift`, `lift_precise`,
`action_rate_l2`, `joint_pos_limits`, `joint_vel_hinge`. Vision tasks
keep the same critic obs as `Cube` so the value function still has
ground-truth ee/cube/goal deltas while the actor learns from pixels.

Defaults (`play=True` mode): `num_envs=4`, infinite episode, no obs
corruption, no curriculum, 4 s `lift_height` resampling. Train mode:
`num_envs=4096`. Override on the CLI with `--env.scene.num-envs N`.

`Play` exists as the minimal idle env — robot + cube + plane, no
rewards, no terminations, no resets beyond timeout. Useful baseline for
viewer / wrapper / dataset-replay work that shouldn't depend on the
manipulation reward stack.
