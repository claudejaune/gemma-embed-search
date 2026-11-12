# Server setup

The script talks to an HTTP embeddings endpoint. It needs three things from any server:

1. `POST /v1/embeddings` accepting `{"model", "input", "encoding_format": "float"}`.
2. Multimodal input parts inside `input`: `{"content": [{"type": "image_url", "image_url":
   {"url": "data:image/jpeg;base64,..."}}]}` for images and
   `{"content": [{"type": "input_audio", "input_audio": {"data": "<base64>"}}]}` for audio.
   Plain strings carry text.
3. `GET /models` listing the model names the router accepts (used by `doctor`).

llama.cpp's `llama-server` satisfies all three. Other OpenAI-compatible servers may accept
the same shape; test with `doctor` and one image before indexing a corpus.

## Servers that do not qualify

**LM Studio fails requirements 2 and 3.** Its `/v1/embeddings` takes only a string or an array
of strings and rejects content parts with `'input' field must be a string or an array of
strings`, so text embeds and images and audio do not. Text-only search works there; the image
and audio features do not, and no client-side change fixes it.

## Starting llama-server

Point one server at the embedding model, with its vision projector and mean pooling:

```bash
llama-server \
  --model  /path/to/embedding-model.gguf \
  --mmproj /path/to/mmproj-embedding-model.gguf \
  --embedding --pooling mean \
  --ctx-size 8192 --batch-size 2048 --ubatch-size 2048 \
  --host 127.0.0.1 --port 1235
```

| Flag | Why it matters here |
| --- | --- |
| `--embedding` | Enables `/v1/embeddings`. Without it every request 404s. |
| `--pooling mean` | Multimodal embedding models are trained for mean pooling. Other pooling makes image-to-text similarity collapse. |
| `--mmproj` | Loads the vision (and, for audio-capable models, audio) encoder. Text works without it, images silently do not. |
| `--ctx-size` | Must hold a whole batch. EmbeddingGemma 2 spends about 1024 vision tokens per image at `-ub 2048` (llama.cpp requests the encoder's maximum 1120-token budget, capped to fit one ubatch), so 8192 tokens fits roughly 7 images per request. Keep `--batch` at or below that, or raise `--ubatch-size` to fit more images. |
| `--api-key` | Optional. If set, pass the same key to the script with `--api-key` or `EMBED_SEARCH_API_KEY`. |
| `--models-preset` / multi-model routers | The `--model` name the script sends must match the name the server advertises in `GET /models`, which is not always the file name. With `-hf`, the advertised name is the `-hf` spec itself, e.g. `ggml-org/embeddinggemma-2-GGUF:Q8_0`. |

With `-hf repo:QUANT` you can skip `--mmproj` entirely: llama.cpp downloads the projector
whose quant matches the model's tag from the same repo (this is `--mmproj-auto`, on by
default). Passing an explicit `--mmproj` path suppresses the auto-download.

**iGPUs and CPU-only machines:** on Intel integrated graphics, the default device selection
offloads to Vulkan, and the i915 driver resets the GPU under sustained image load (a compute
dispatch outlives the kernel's preemption timeout → `VK_ERROR_DEVICE_LOST` → the server
aborts). It is intermittent: a small pilot can pass while a long index dies. Use
`--device none -ngl 0` on iGPU or CPU-only hosts; partial offload (`-ngl` below the layer
count) is sometimes faster but still crash-prone, so CPU is the reliable baseline.

A model that is loaded on demand can be evicted when another model is requested. The next
embedding call reloads it and costs a few extra seconds; nothing is lost.

## Image size and the token budget

There is no maximum image dimension. The server pins the number of vision tokens per image
(`--image-max-tokens` and `--image-min-tokens`) and resamples whatever you send to that patch
grid. Measured on one photo with a 280-token budget:

| Sent as | Request body | Vision tokens | Cosine vs full-resolution vector |
| --- | --- | --- | --- |
| 216x384 | 39 KB | 303 | 0.923 |
| 431x768 | 119 KB | 303 | 0.936 |
| 575x1024 | 190 KB | 303 | 0.955 |
| 862x1536 | 366 KB | 268 | 0.996 |
| 2300x4096 (original) | 1.8 MB | 268 | 1.000 |
| 7200x12000 (upscaled) | 8.8 MB | 256 | 0.999 |

What that means:

- An oversized image does not fail and does not overflow the context. It is downsampled. The
  only costs are upload bytes and decode time: 0.21 s at 768 px against 0.97 s at 12000 px.
- Detail finer than the grid is unrecoverable at any resolution. About 280 tokens is a
  20x14 patch grid, so a face occupying 30 pixels of a 4000-pixel photo is invisible to the
  model no matter how sharp the file is. Do not chase small objects by sending bigger files.
- Downscaling twice costs accuracy: 768 px lands at 0.936 cosine against the full-resolution
  vector, while 1536 px lands at 0.996. That is why the default is 1536. Use `--resize 0` to
  send files untouched, or `--resize 512` for a fast first pass over a huge folder.
- The emitted token count drifts around the configured budget (256 to 303 for a 280 setting),
  so leave headroom when sizing `--batch` against `--ctx-size`.
- The budget is clamped by the ubatch size: the server logs `cap image_max_tokens
  (original=1120) to 1024 (n_ubatch = 2048)` when `-ub` is smaller than the requested
  budget. At the capped 1024, an 8192-token context holds only about 7 images per request —
  a larger `--batch` overflows the context and the request hangs until it times out.

In the model's own terms, an image is compressed to "soft tokens": trained at a default of
280, supported from 70 to 1120, with the model card noting that a higher budget buys better
fine-grained detail. llama.cpp already requests the maximum (1120, capped by
`--ubatch-size`), so there is nothing to gain in quality by raising `-ub` further — a larger
`-ub` only fits more images per request.

## Choosing a model

Any multimodal embedding model that maps text and images into one vector space works.
EmbeddingGemma 2 (740M parameters, 768-dim output, text plus image plus audio, Apache-2.0)
is the reference target for this skill and is what the numbers in these docs were measured
on. Requirements:

- A GGUF build current enough to know the architecture: the `gemma-embedding2` architecture
  landed in llama.cpp on 2026-10-06, so v0.9.0 or any later nightly works, and older builds
  stop with "unknown model architecture" or "unsupported". Update the server rather than the
  script.
- Use Q8_0 or Q6_K quantization. Do not run these models in F16: activations overflow and
  the vectors come out NaN or meaningless.
- Keep the same quantization for indexing and querying. Mixing them degrades similarity
  without any error.

## Verifying

```bash
uv run scripts/embed_search.py --url http://127.0.0.1:1235 --model <name> doctor
```

A healthy run reports `"reachable": true`, lists the model in `models`, and returns
`"text_embedding": {"ok": true, "dim": 768}`. Then prove vision end to end: index two or
three pictures and search for something visible in one of them. Text passing and images
failing is the most common half-broken setup, usually a missing `--mmproj`.

## Where state lives

| Path | Contents |
| --- | --- |
| `~/.config/gemma-embed-search/config.json` | url, model, api key, default root. Mode 0600. Delete it to reset. |
| `~/.cache/gemma-embed-search/<folder>-<hash>.jsonl` | The index. One JSON object per file: path, mtime+size key, modality, packed float32 vector, 24-bin color histogram. |

The index is plain JSON lines: `wc -l` counts entries, `jq` reads them, and deleting it is
always safe. It costs about 4 KB per embedded file.

The color histogram is 24 bins (15° each) and counts only saturated pixels. An index built
with the older 12-bin histogram still embeds and searches, but its color ratios are coarser
and over-count warm tones; `search --color` warns when it sees one. Run
`index --refresh-colors` to recompute the histogram in place from the files — local only, no
server, seconds to run, and the embeddings are untouched.
