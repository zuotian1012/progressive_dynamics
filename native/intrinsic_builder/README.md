# Intrinsic hierarchy preprocessing adapter

This separate command-line executable calls the public implementation of
**Surface Multigrid via Intrinsic Prolongation**, Liu, Zhang, Ben-Chen and
Jacobson (2021), which is the intrinsic mapping work referenced by PSQ.

- Source: https://github.com/HTDerekLiu/surface_multigrid_code
- Pinned commit: `adc49abe10a97200a3a5735bd370e98560091a37`
- Its libigl submodule: `70d2abaa53616af35c7857bf8f52c8a234f150ff`
- Eigen: 3.4.0.
- Original license: GPL-3.0, preserved in `third_party/surface_multigrid/LICENSE`.
- This linked adapter is marked GPL-3.0-or-later. No project-wide license for
  unrelated source files or the shirt assets is implied.

We add a headless exporter and a probabilistic quadric adapter, retaining joint
self-parameterization and the exported intrinsic barycentric maps and containing
coarse triangle indices. These maps are not nearest-point interpolation.
`SSP_qslim_probabilistic.cpp` is an adapted copy of the author's GPL source;
the external checkout remains unchanged. The linked target uses this copy in
place of the original `SSP_qslim.cpp`.

The default metric adds uncertainty to each classical face plane using
[Trettner and Kobbelt's reference implementation](https://github.com/Philip-Trettner/probabilistic-quadrics).
Its MIT license is retained in `third_party/probabilistic_quadrics`.
The downloaded branch archive is identified by the reference header SHA-256
`3e80a263705920b6900bda815887119fa989ada04fe14991127d7bd073a9365c`,
not an asserted pinned Git commit. Classical boundary terms and the original
small regularization remain. Normal standard deviation defaults to 0.01 and
position standard deviation to 0.01 times mean triangle edge length. These are
explicit local settings, not recovered author scene parameters. Use
`--quadric ordinary` for the historical metric.

For displacement-controlled plate experiments, `--lock-boundary` forbids
collapses incident on any original boundary vertex. This optional mode retains
the boundary polygon and vertex positions (up to OBJ serialization precision),
but can prevent reaching aggressive target counts. It is off by default and
requires a rebuilt executable. The plate demo uses a separate binary under
`output/intrinsic_plate_build`, leaving historical binaries unchanged.

The Python solver uses the exported hierarchy and implements PSQ's nonlinear
normal/tangent offsets separately. The original finest topology is retained.
The executable exports reversible patch histories in `expansion_N.json`.
Python preparation validates their full oriented topology/rest reconstruction,
then saves `expansion_i_j.json` and dependency-color statistics with the assets.
The runtime CPU safe expansion lives in `progressive/safe_expansion.py`; see
[`docs/SAFE_EXPANSION.md`](../../docs/SAFE_EXPANSION.md) for scope and validation.

Release build (dependency source is included in `third_party/`):

```powershell
cmake -S native/intrinsic_builder -B build/intrinsic -G "Visual Studio 17 2022" -A x64
cmake --build build/intrinsic --config Release --parallel 2
build/intrinsic/Release/intrinsic_builder.exe --self-test
python -m progressive.prepare_intrinsic --input tests/fixtures/intrinsic_expansion/mesh_0.obj --targets 12 --output output/my_intrinsic_assets
```

The following results describe the original experiment workspace, not bundled data:

The current probabilistic two-level shirt (`output/shirt_paper_2level`) has
533/4090 vertices and 1000/8000 faces, nonzero PSQ offsets, unchanged finest
topology, and passes rest self-intersection and 0.2 mm clearance checks.
Ordinary mode reproduces the historical 534/4090 hierarchy and U exactly.

The probabilistic three-level attempt (`output/shirt_paper_3level`) has no rest
self-intersections but its middle layer has only 0.017519 mm clearance, below
0.2 mm. It is explicitly marked infeasible. The historical ordinary three-level
attempt had actual intersections. These are distinct failure conditions.
Inspect BOTH `nonintersecting_rest_hierarchy` and
`thickness_feasible_rest_hierarchy`. Per-level null
`minimum_distance_below_clearance_m` means no pair inside the checked clearance,
not a global infinite distance. Preprocessing reports these failures without
repairing them; simulation additionally preflights cloth and obstacle geometry
at every level before stepping any level.

Inputs and outputs are local files. Nothing is uploaded or published.
