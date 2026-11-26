# Third-party source and asset notices

This is a local release-preparation snapshot. No repository-wide license has been selected here. Existing upstream licenses and source headers continue to apply; this inventory is not a complete licensing review.

- **libuipc**: source preparation pins `9c5710ee4b716502e3757e7312360e5ee68cb79f`. Final effective modified source is included under `third_party/libuipc/`; historical reference/modified sources also exist under `native/`. The upstream LICENSE and NOTICE found in the local source tree are preserved under `third_party/notices/libuipc-*`. The build includes additional dependencies whose notices must be reviewed before binary distribution.
- **surface_multigrid_code**: `native/intrinsic_builder/SSP_qslim_probabilistic.cpp` identifies commit `adc49abe10a97200a3a5735bd370e98560091a37` and carries `SPDX-License-Identifier: GPL-3.0-or-later`. The local upstream license text is preserved as `third_party/notices/surface-multigrid-LICENSE.txt`.
- **genesis-ipc-shirt-demo / Scene527 assets**: the asset fetcher names commit `c29cc92f954fa4b803fb5cfe3155d6d9dbd5b088`. The local upstream NOTICE explicitly states no open-source license has been selected and that bundled assets retain upstream terms. Its text is preserved as `third_party/notices/shirt-assets-NOTICE.md`; no scene asset bundle is included in this snapshot.
- **Eigen 3.4.0**: required hierarchy-builder headers and their `COPYING.*` notices are included under `third_party/eigen/`. The libuipc xmake build resolves its own Eigen dependency separately.
- **libigl**: hierarchy headers from `70d2abaa53616af35c7857bf8f52c8a234f150ff` are included under `third_party/libigl/`, with `LICENSE.MPL2` and `LICENSE.GPL` preserved.
- **probabilistic-quadrics**: reference headers and MIT license are included under `third_party/probabilistic_quadrics/`. The original reference header SHA-256 is `3e80a263705920b6900bda815887119fa989ada04fe14991127d7bd073a9365c`; this snapshot does not assert an unverified Git commit.
- **pybind11 v3.0.4**: header source and its license are included under `third_party/libuipc/vendor/pybind11/`.
- **Other native dependencies**: libuipc also uses xmake-resolved packages and embedded METIS/GKlib code. Existing notices in the vendored tree are retained. Binary redistribution requires carrying the licenses of the resolved binary dependencies.
- **Historical alternative backends**: Codim-IPC and libshell remain referenced by legacy scripts, but are not dependencies of the new unified CUDA build and are not vendored here.
- **Papers**: local PDFs were excluded; citation metadata and links will be added to the final research documentation.

This inventory identifies the sources supplied for the current build; historical reference code and scene assets still require scope checks before a public release.
