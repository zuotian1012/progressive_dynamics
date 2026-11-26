# Intrinsic expansion regression fixture

Generated locally from `cloth.mesh.grid(4)` (25 vertices, 32 faces), with
`z = 0.08*sin(4*x)*cos(3*y)`, then simplified to 12 faces / 11 vertices.
No external shirt asset is embedded in this fixture.

The exporter uses the pinned Liu et al. implementation and the explicit local
probabilistic quadric settings documented in `native/intrinsic_builder/README.md`.
`expansion_1.json` contains the 14 successful collapses, their original vertex
identities, oriented finite patch faces, joint UVs and pre/post rest positions.
`map_1.txt` and `anchors_1.txt` are the independently exported level map.

Tests verify reverse topology/rest reconstruction, disjoint dependency batches,
rest recovery, actual self-contact under deformation, bounded seeded sampling,
per-pair ACCD ratios, continuous triangle area checks and strain correction.
