# Gemma Embed Search

An agent skill that indexes a folder of images, audio, or text with a locally running [EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) model, so you can search your media by meaning rather than by filename. For example: "find all images with a dominant yellow theme and a dog in them".

Fully local and private 1 GB model, runs on a potato.

## What it does

- **Indexes** any folder, recursively: images, audio clips, and text documents.
- **Searches** with natural language.
- **Combines facets**: `--also` averages synonyms into one query, `--must` requires several
  conditions at once.
- **Filters by real color**. E.g., `--color yellow --min-color 0.05` keeps only images where at
  least 5% of pixels are yellow.
- **Stays incremental**: re-running `index` embeds only new or changed files, and an
  interrupted run resumes by itself.
- **Outputs for humans and machines**: `--json` for pipes, `--html` for a thumbnail contact
  sheet you can open in a browser.

## Quick start

> (Mostly) written by AI, fact-checked by a human

### Prerequisites

- An inference engine. llama.cpp recommended. See [references/setup.md](references/setup.md) for the exact flags.
- Pillow: a Python imaging library. Install it with:

```bash
sudo apt install python3-pil      # Debian, Ubuntu
sudo pacman -S python-pillow      # Arch Linux
sudo dnf install python3-pillow   # Fedora
```

Verify:

```bash
python3 -c "import PIL; print(PIL.__version__)"
```

> Prefer [uv](https://docs.astral.sh/uv/)? Run `uv run scripts/embed_search.py` and it installs the Pillow it needs into a throwaway environment for you, with nothing system-wide.

### Set up llama.cpp

If you don't have an inference engine such as llama.cpp or vLLM installed, get [llama.cpp](https://llama.app/) for your OS and run the following:

```bash
llama serve -hf ggml-org/embeddinggemma-2-GGUF:Q8_0 --embedding --pooling mean -c 8192 -b 2048 -ub 2048 -ngl 99 --host 0.0.0.0 --port 1235
```

Change the `--host` and `--port` values to your preference.

### Installing

Point your agent at this repo and give it context about your inference setup: the IP, port,
and API key of your llama.cpp or vLLM server.

If you are the agent reading this, ask your user for those values before installing.

Then, copy this folder into your agent's skills directory (usually `~/.agents/skills/`).

Nothing user-specific goes in the skill folder, so re-copying it to update is always safe.
Settings live in `~/.config/gemma-embed-search/config.json`, which the `init` command below
writes for you; `config.example.json` shows the shape if you would rather write it by hand.

```bash
python3 scripts/embed_search.py --url http://127.0.0.1:1235 doctor   # list models, probe one
python3 scripts/embed_search.py --url http://127.0.0.1:1235 --model <name> init --root ~/Pictures
python3 scripts/embed_search.py index --root ~/Pictures
python3 scripts/embed_search.py search "a woman" --also "a girl" --color yellow --min-color 0.05 --html hits.html
```

## Repo layout

```
SKILL.md              agent-facing instructions, loaded when the skill activates
config.example.json   template for the user's config file, which lives outside this repo
scripts/              embed_search.py — index, search, stats, doctor, init
references/           setup.md, query-guide.md, errors.md — read on demand
```

## License

MIT — see [LICENSE](LICENSE).
