---
name: gemma-embed-search
description: Index a folder of images, audio, or text with a local multimodal embedding model and search it in natural language. Builds a resumable vector index through any OpenAI-compatible /v1/embeddings server that accepts multimodal content parts, such as llama.cpp's llama-server running EmbeddingGemma 2, then ranks files by semantic similarity with color filtering, multi-facet AND queries, JSON output, and HTML contact sheets. Use when the user wants to find photos or media by what is in them ("every photo of my dog", "shots with a red car at night", "portraits in yellow"), organize or triage a large media folder, dedupe near-identical images, or run semantic search over local files without sending them to a cloud API.
compatibility: Requires Python 3.9+ with Pillow (or `uv run`), plus a running embeddings server that accepts multimodal content parts. No internet access needed once the model is local.
license: MIT
metadata:
  version: "1.0.0"
---

# Gemma Embed Search

Retrieve local files by meaning. One embedding space holds images, audio, and text, so a
typed query returns the pictures, clips, or documents that match it. The model runs on the
user's hardware through a local server; nothing is uploaded anywhere.

`scripts/embed_search.py` does four things: `doctor` checks the server, `init` saves the
connection, `index` embeds files into a cache, and `search` ranks them against a query.

## Local configuration

<!-- BEGIN LOCAL CONFIG -->
| Setting | Value |
| --- | --- |
| Server URL | `http://127.0.0.1:8080` |
| Model | *not set — run `doctor` to list* |
| API key | *none* |
| Default folder | *not set — pass `--root`* |
<!-- END LOCAL CONFIG -->

An installing agent replaces this table with the user's values and writes the same values
to `~/.config/gemma-embed-search/config.json`, so later commands need no connection flags.
Ask the user for anything you do not know; never guess a model name, read it from `doctor`.

## Setup checklist

Progress:
- [ ] Step 1: Confirm the server and pick a model name (`doctor`)
- [ ] Step 2: Save the connection (`init`)
- [ ] Step 3: Index a 20-file pilot and eyeball one query
- [ ] Step 4: Index the whole folder
- [ ] Step 5: Answer the user's real query, with `--html` so they can see the hits

If the Local configuration table already shows real values, skip Step 2.

Run the script with `uv run scripts/embed_search.py` (installs Pillow from the script's
inline metadata) or `python3 scripts/embed_search.py` if Pillow is already importable.

**Step 1 — find the server and model name.** Never guess the model name; read it from the
server.

```bash
uv run scripts/embed_search.py --url http://127.0.0.1:8080 doctor
```

`doctor` prints the models the server offers and probes one embedding. If the user has not
started a server yet, read [references/setup.md](references/setup.md) and give them the
command for their model.

**Step 2 — save it** so later calls need no flags. Pass `--api-key` only if the server
requires one; it is written to the config file with mode 0600.

```bash
uv run scripts/embed_search.py --url http://127.0.0.1:8080 --model <name-from-doctor> init
uv run scripts/embed_search.py --root ~/Pictures index --limit 20
```

**Step 3 — pilot before the long run.** Indexing is the expensive step: roughly 0.25 s per
image on a desktop GPU, so 10,000 files is about 40 minutes. Index 20 files, run one query,
and confirm the results look right before committing to the full pass.

**Step 4 — full index.** Re-running `index` only embeds new or changed files, so an
interrupted run resumes on its own.

```bash
uv run scripts/embed_search.py --root ~/Pictures index
```

**Step 5 — search.**

```bash
uv run scripts/embed_search.py search "a black labrador on a beach" --top 20 --html hits.html
```

## Query cookbook

| Intent | Command |
| --- | --- |
| Plain semantic search | `search "a crowded night market"` |
| Two facets that must both hold | `search "a woman" --must "the color yellow"` |
| Synonyms so one phrasing does not dominate | `search "a woman" --also "a girl" --also "female portrait"` |
| Subject plus a real color constraint | `search "a woman" --color yellow --min-color 0.05` |
| Mostly monochrome or dark frames | `search "any image" --max-color blue --modality image` |
| Only one kind of file | `search "siren sounds" --modality audio` |
| Machine-readable for further work | `search "..." --json --top 50` |
| Faster, slightly less accurate scoring | `search "..." --dims 256` |

`--also` averages paraphrase vectors into one query. `--must` keeps the score of the
weakest facet, which is how you express AND. Use `--also` for synonyms and `--must` for
independent conditions.

## Reading the output

Plain output is `score [modality] path`, best first. Color runs insert the pixel ratio
after the score. `--json` emits `{"query", "index", "total", "filtered_out", "results":
[{"rank", "score", "color", "modality", "path"}]}` on stdout; all progress goes to stderr,
so stdout always pipes cleanly into `jq`.

Cosine scores from this model family sit between about 0.3 and 0.8. The absolute number
means little; the gap between hits and misses matters. If the top of the list is wrong, the
query is wrong — rephrase it rather than trusting the ranking.

For visual review, always offer `--html`: it writes a contact sheet of thumbnails with
scores and filenames that the user can open in a browser. A bare `--html` writes `hits.html`
in the working directory; `--html FILE` sets the path. Thumbnails are embedded as base64, so
the sheet is self-contained and travels on its own. They are 512 px wide by default; raise
`--thumb-width` for close inspection of detail.

## Gotchas

- A color word inside the query text is a soft hint, not a filter. `"a woman with a yellow
  theme"` ranks a woman in an orange jacket first. Use `--color yellow --min-color 0.05`
  for an actual constraint; it is computed from pixels, not from the model.
- `--color` needs the index to hold color stats, which only `index` computes. An index
  built with an older version reports 0.0 for everything: re-index with `--force` if color
  filtering silently excludes every file.
- Index and query must use the same model. Switching `--model` silently mixes incompatible
  vectors; re-index with `--force` after any model change and note the model in `stats`.
- The server must run the embedding model with mean pooling and its vision projector
  (`--mmproj`) loaded. Without the projector, image requests fail or return a vector that
  ignores the picture. `doctor` proves text works; run one image query to prove vision.
- Images are re-encoded to JPEG at longest-side 1536 before they are sent. The re-encode is
  required, not cosmetic: servers commonly reject WebP, 16-bit, or mislabeled files, and
  Pillow reads those instead. The model has no pixel limit — the server resamples every image
  to a fixed vision-token budget — so `--resize` trades upload size for fidelity, not
  correctness. Measured numbers and the details are in
  [references/setup.md](references/setup.md#image-size-and-the-token-budget).
- HEIC and raw camera formats are not read unless Pillow was built with the plugin. The
  script logs `skip <file>: cannot identify image file` and continues; check
  `skipped_errors` in the index summary rather than assuming full coverage.
- `search` on a folder that was never indexed exits 4 with a message. Do not treat an empty
  result list as "no matches" until the index count in `stats` looks right.
- Audio is capped by the server context: about 5.5 minutes per clip at 25 tokens per
  second. Longer files fail as a request error, not as a bad match.
- Scores are not comparable across corpora or across quantizations. Never promise a
  threshold; tune `--min-color` and `--top` against what the user sees.

## Script reference

```text
doctor                       Check the server, list models, probe an embedding
init [--root DIR]            Save url, model, key, and default root to the config file
index   --root DIR           Embed new and changed files (incremental, resumable)
        [--modality auto|image|audio|text] [--include '*.png,sub/*.jpg']
        [--exclude-dir node_modules,.git] [--limit N] [--batch 16] [--resize 1536]
        [--force] [--compact]
search  "query"              Rank indexed files; prints score, modality, path
        [--also P]... [--must P]... [--color NAME --min-color F | --max-color F]
        [--modality image] [--dims 512|256|128] [--top 20] [--offset 0]
        [--json] [--html [FILE]] [--thumb-width 512]
stats                        Entry count, modality mix, model, index size
```

Exit codes: `0` success, `2` usage error, `3` server unreachable or auth rejected,
`4` index missing, `5` nothing to do, `6` server error mid-run. Run
`scripts/embed_search.py <command> --help` for full flag documentation.

Config resolution is flag > environment (`EMBED_SEARCH_URL`, `EMBED_SEARCH_MODEL`,
`EMBED_SEARCH_API_KEY`, `EMBED_SEARCH_ROOT`) > config file at
`~/.config/gemma-embed-search/config.json`. The index defaults to
`~/.cache/gemma-embed-search/<folder>-<hash>.jsonl`, one JSON record per file, so it is
safe to keep several indexes for different folders.

## Further detail

- [references/setup.md](references/setup.md) — starting a server, model and flag requirements, what `doctor` means
- [references/query-guide.md](references/query-guide.md) — query construction, facet logic, color thresholds, tuning against real results
- [references/errors.md](references/errors.md) — every exit code and the fix for each failure mode
