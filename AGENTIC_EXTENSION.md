# GPU and iterative development extension

This local branch extends the published WorldSolver checkout at
`f0734148a0fe028b8f09ce4cf9834bac467b3045`.

- Luisa path tracing is the default rendering backend. Pass `--renderer pyrender`
  to retain the original backend. Luisa uses the independent native runtime
  named by `LUISA_RENDER_BIN`; it does not import Genesis.
- The isolated controller in `../worldsolver_runtime/driver/run_iterative.py`
  gives Codex access to one assigned GPU and a host-owned budget of at most
  20 full simulation attempts per task. Numerical diagnostics and video
  feedback support iterative revisions. Final evaluation reuses a tested
  source snapshot and its existing artifacts.
- Tasks 169–171 are additional Text2Sim cases. Their frozen assets, initial
  conditions, material laws, controls and source hashes are included. They
  are not part of the original 168-task results.

The controller, generator and evaluator have separate environments under
`../worldsolver_runtime`. Generated runs live outside this checkout under
`/data/xiongxiaoyu/workspaces/worldsolver/iterative/`. The original Genesis
environment, source scenes and previous WorldSolver run are preserved.

Setup details, protocol differences, test evidence and current results are in
[the extension report](../worldsolver_runtime/UPGRADE.md). A successful execution
and video do not by themselves establish physical correctness or scene fidelity.
