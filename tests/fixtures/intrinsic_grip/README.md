# Split endpoint identity regression

Own 8 cm planar grid (25 vertices, 32 faces), reduced by the pinned native
probabilistic-quadric/self-parameterization builder to 10 vertices / 11 faces.
The saved first coarse step and physical box-jaw obstacle geometry reproduce
`output/vertical_grip_strong_infeasible` (dt .01, contact stiffness 1e4).
No original shirt or robot asset is embedded here.

The old coarse storage slots include 9 split parents and one unchanged vertex
(24). Freezing all 10 after refinement gives four impossible fixed edges;
retiring split parent identities leaves vertex 24 and every obstacle fixed,
while allowing strain correction to repair both newly created endpoints.

Maps, oriented patches, UVs and rest geometry are independent native exports.
