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
strings`, so text embeds and images and audio do not. It also serves `/v1/models` but not
`/models`, and answers `GET /models` with HTTP 200 and an error body — which makes `doctor`
report `"reachable": true, "models": []` instead of an error. Text-only search works there;
the image and audio features do not, and no client-side change fixes it.

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
| `--ctx-size` | Must hold a whole batch. At about 280 vision tokens per image, 8192 tokens fits roughly 25 images per request. |
| `--api-key` | Optional. If set, pass the same key to the script with `--api-key` or `EMBED_SEARCH_API_KEY`. |
| `--models-preset` / multi-model routers | The `--model` name the script sends must match the name the server advertises in `GET /models`, which is not always the file name. |

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

## Choosing a model

Any multimodal embedding model that maps text and images into one vector space works.
EmbeddingGemma 2 (740M parameters, 768-dim output, text plus image plus audio, Apache-2.0)
is the reference target for this skill and is what the numbers in these docs were measured
on. Requirements:

- A GGUF build current enough to know the architecture. If the server exits with an
  "unknown model architecture" or "unsupported" error, update the server rather than the
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
| `~/.cache/gemma-embed-search/<folder>-<hash>.jsonl` | The index. One JSON object per file: path, mtime+size key, modality, packed float32 vector, color histogram. |

The index is plain JSON lines: `wc -l` counts entries, `jq` reads them, and deleting it is
always safe. It costs about 4 KB per embedded file.
