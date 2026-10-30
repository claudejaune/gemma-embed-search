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

## Starting llama-server

Point one server at the embedding model, with its vision projector and mean pooling:

```bash
llama-server \
  --model  /path/to/embedding-model.gguf \
  --mmproj /path/to/mmproj-embedding-model.gguf \
  --embedding --pooling mean \
  --ctx-size 8192 --batch-size 2048 \
  --host 127.0.0.1 --port 8080
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
uv run scripts/embed_search.py --url http://127.0.0.1:8080 --model <name> doctor
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
