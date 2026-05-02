# Models/

Core ML model packages live here. **This directory is intentionally empty in
git** — its contents are downloaded by `scripts/fetch-models.sh` from
[huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml).

After fetching, expect files like:

```
kokoro_duration_t128.mlpackage/
kokoro_f0ntrain_t240.mlpackage/
kokoro_f0ntrain_t560.mlpackage/
kokoro_f0ntrain_t800.mlpackage/
kokoro_f0ntrain_t1200.mlpackage/
kokoro_f0ntrain_t2400.mlpackage/
kokoro_decoder_pre_3s.mlpackage/
kokoro_decoder_pre_7s.mlpackage/
kokoro_decoder_pre_10s.mlpackage/
kokoro_decoder_pre_15s.mlpackage/
kokoro_decoder_pre_30s.mlpackage/
kokoro_decoder_har_post_3s.mlpackage/
kokoro_decoder_har_post_7s.mlpackage/
kokoro_decoder_har_post_10s.mlpackage/
kokoro_decoder_har_post_15s.mlpackage/
kokoro_decoder_har_post_30s.mlpackage/
```

This whole directory is referenced as a **folder reference** in the Xcode project
(see `project.yml`), so its contents are copied verbatim into the .app bundle
and not pre-compiled by Xcode. `KokoroPipeline` compiles each `.mlpackage` to
`.mlmodelc` at runtime on first access.

To re-run the download:

```
bash scripts/fetch-models.sh
```
