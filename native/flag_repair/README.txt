Isolated flag L0 non-descent rescue

This directory is deliberately separate from native/libuipc and the installed
or default uipc runtime. No existing shirt entry point imports this variant.

The source copies retain the existing model and safety checks. The changes are:
- psq_exact_offset.cuh: exact VJP of the normal and polar-rotation offsets.
- psq_proxy.cu: full offset derivative opt-in and single-assembly retry opt-in.
- psq_preview_stop.inl: on verified first-Newton non-descent, request one retry.
- advance_ipc.cu: include the local preview block and independently require/log
  full energy descent for that retry. It restores and rejects a failed retry.
- check_exact.cu: exports the actual derivative kernel for independent testing.

The retry is a robustness extension, not the original paper's frozen-offset
approximation. Original coarse non-descent stopping resumes after the retry.
It is validated only for this flag L0; no multilevel accuracy claim is made.

Build from the project root:
  .venv/Scripts/python.exe scripts/build_flag_repair.py --variant rescue

The build queries xmake's existing compiler/linker metadata, compiles only the
two local translation units, and relinks read-only cached objects into
output/flag_L0_repair_20261118/rescue_runtime/src. It never runs the shared
xmake build/install and never writes the default DLL or common source tree.
The configured shared source build and its object files must already exist.
Build provenance records the input objects and output DLL hashes.

Run (choose a new output directory):
  .venv/Scripts/python.exe -u scripts/run_flag_l0_repair.py --scene flag --assets output/pdpp_flag_paper_assets_20261117_v2 --output output/flag_L0_rescue_NEW --native-package output/flag_L0_repair_20261118/rescue_runtime/src --steps 500 --rescue

Capture stdout/stderr to the sibling output/flag_L0_rescue_NEW.log to retain
the per-event energy and non-descent audit evidence.
The original scripts/run_paper_l0.py and all shirt commands remain unchanged.

Validated result, controls, limits, and media:
  output/flag_L0_repair_20261118/RESULTS.txt
