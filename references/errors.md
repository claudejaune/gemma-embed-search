# Errors and fixes

## Exit codes

| Code | Meaning | First move |
| --- | --- | --- |
| 0 | Success | — |
| 2 | Usage error: bad flag, missing `--root`, unreadable config | Read the message; it names the flag and the expected form |
| 3 | Server unreachable, auth rejected, or request refused | Run `doctor` |
| 4 | Index missing or empty | Run `index --root DIR` |
| 5 | Nothing to do | Normal when the index is already current |
| 6 | Server answered but the response was unusable | Check that the model is an embedding model, not a chat model |

## Connection and auth

**`cannot reach http://... (Connection refused)`** — the server is not running or the port is
wrong. Start it (see setup.md) or pass `--url`. Confirm the port with the user: containers
and port forwards often publish a different port than the one the server listens on.

**`server rejected the credentials (HTTP 401)`** — the server was started with an API key.
Set `EMBED_SEARCH_API_KEY` or pass `--api-key`, then `init` to store it.

**`HTTP 404 ... Check --model matches a model the server knows`** — the name sent does not
exist on that server. `doctor` lists the valid names; routers often expose a preset section
name or an alias rather than the GGUF file name.

**`unexpected response, expected a 'data' list`** — the endpoint is not an embeddings
endpoint, or the loaded model is a generative model that ignores `--embedding`. Load an
embedding model.

## Indexing

**`nothing to do: N entries already current`** — expected. The key is path plus mtime plus
size, so untouched files are never re-embedded. Force a rebuild with `--force`.

**`skip <file>: cannot identify image file`** — corrupt download, zero-byte file, or a
format Pillow cannot decode (HEIC and camera RAW need extra plugins). The run continues;
check `skipped_errors` in the summary to see how many were lost.

**Request errors or a context-overflow log line** — the batch does not fit the server
context, not the image. One image costs a fixed ~280 vision tokens whatever its pixel size,
so `--batch 16` needs roughly 4,500 tokens of context. Lower `--batch`, or raise
`--ctx-size` on the server. A single enormous file only costs bandwidth and decode time:
a 7200x12000 input took 0.97 s against 0.21 s for a 768 px version, and still returned a
vector within 0.001 cosine of the full-resolution one.

**`timed out after 300s`** — a big batch on a slow device, or the server is busy generating
with another model. Raise `--timeout`, lower `--batch`, or give the embedding model its own
server slot.

**Index grows but results stay wrong** — the vectors are fine but were produced by a
different model than the query. `stats` prints the model recorded at index time; if it
differs from the current `--model`, re-index with `--force`.

**Vectors are all zeros or NaN** — the model is running in F16 or with the wrong pooling.
Use Q8_0 or Q6_K with `--pooling mean`.

## Searching

**`no index at ...`** — pass `--root` for the folder you want, or `--index` for a specific
file. The default index path is derived from the root path, so a moved folder gets a fresh
empty index until you point at the old one or re-index.

**Zero results with a color filter** — either the threshold is too high or the index has no
color stats. Rerun without `--color` to confirm matches exist, then lower `--min-color`. If
every file reports `color: 0.0`, the index predates color statistics: rebuild with
`--force`.

**Results look unrelated to the query** — check three things in order: the index is not empty
(`stats`), the model matches the one used at index time, and the query is concrete. "nice
photos" retrieves noise; "a red bicycle leaning on a wall" retrieves the right thing.

**Old entries for deleted files still appear** — run `index --compact` to drop records whose
file is gone.

## Performance

Rough desktop-GPU cost is 0.25 s per image plus about 5 ms for color stats, so a 10,000-file
corpus is around 45 minutes. To go faster: `--resize 512`, larger `--batch`, or `--dims 256`
at query time. To go faster on repeat runs, do nothing: indexing is incremental by design.
