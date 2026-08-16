# Third-party dependency versions

This project vendors source snapshots for small libraries and downloads official
binary packages for Embree and Open Image Denoise. Rolling upstream projects are
pinned to an exact commit so builds remain reproducible.

## Updated in August 2026

| Dependency | Version / revision | Integration notes |
| --- | --- | --- |
| stb_image | 2.30, upstream commit `2c980bb59875b0d32144a71867fbdebb2f77cd20` | Vendored as `stb_image/stb_image.hxx`; file SHA-256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`. |
| Sokol App | upstream commit `e01e395cd5b916fc381913161244cb5abd3fbc37` | Vendored as `sokol/sokol_app.h`; file SHA-256 `4ace5d4221943950b71dad37bf69ce0449897aee12dab5af1daeeca8fc719487`. Sokol does not publish numbered releases. This is the newest revision immediately before its Metal backend began requiring the macOS 14-only `CADisplayLink` API, preserving this project's macOS 13.3 deployment target. |
| Native File Dialog Extended | 1.3.0, tag commit `fc168e8605bfa51aaec22ab0c4e46b9de665a437` | The project maintains a small platform-only CMake target and a wrapper in `sources/etx/core/platform.cxx`. Source archive SHA-256 `2fea19102cf4d5283a80fb87a784792166988e85bb92baa962d34f72b22dcc1a`. |
| Open Image Denoise | 2.5.0, tag commit `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7` | Official binary package; headers and runtime files under `oidn/` are locally installed and mostly ignored. macOS archive SHA-256 `586142ec125de0bf5b01d3cc4c76985d4fafb0fc91e9f6562e32f3b669f86be5`; Windows archive SHA-256 `6ae0474ef7606d68647c1e2c2842832d6af01128ee84a523d30368a91013b707`. |
| Embree | 4.4.1, tag commit `f590db83ef6559387df7f6d8725c34fb7acf851d` | Downloaded by CI or supplied through `EMBREE_LOCATION`; not vendored. macOS ARM64 archive SHA-256 `12236c79f37e98224582f7fa9e113160b3d4eb5eb5f24846a96faf05e32a5562`; Windows x64 archive SHA-256 `5aa1fa3161a720f9c610b06c652b80f421cb2b1fe8221fdf92b74157336fdf8f`. |

OIDN 2.5.0 and Embree 4.4.1 share the oneTBB 12 runtime. The build deliberately
packages OIDN's newer oneTBB 12.18 copy for both libraries.

## Confirmed current, unchanged

- Dear ImGui 1.91.9
- nlohmann/json 3.12.0
- xxHash 0.8.3
- stb_image_write 1.16

## Deferred updates

- Dear ImGui 1.92 requires a coordinated renderer-backend migration for dynamic
  texture and font-atlas lifecycle handling.
- DXC/SPIR-V tooling, TinyGLTF, TinyEXR, TinyObjLoader, NanoVDB, enkiTS, and
  MicroProfile have broader API, shader, build-system, or runtime implications
  and should be updated as focused changes rather than as part of the low-risk
  dependency batch.
