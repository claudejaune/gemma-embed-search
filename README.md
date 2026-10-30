# Gemma Embed Search

An [Agent Skill](https://agentskills.io) that indexes a folder of images, audio, or text with
a local multimodal embedding model, then lets an agent answer questions like *"find every
photo with a woman in it, yellow theme"* by meaning rather than by filename.

Everything runs on your own machine against a local embeddings server. No files leave it.

## What it does

- **Indexes** any folder: images, audio clips, and text documents land in one vector space.
- **Searches** with natural language, ranked by cosine similarity.
- **Combines facets**: `--also` averages synonyms into one query, `--must` requires several
  conditions at once.
- **Filters by real color**: `--color yellow --min-color 0.05` keeps only images where at
  least 5% of pixels are yellow. Asking for a color in the query text is only a soft hint.
- **Stays incremental**: re-running `index` embeds only new or changed files, and an
  interrupted run resumes by itself.
- **Outputs for humans and machines**: `--json` for pipes, `--html` for a thumbnail contact
  sheet you can open in a browser.

## Quick start

```bash
python3 scripts/embed_search.py --url http://127.0.0.1:8080 doctor   # list models, probe one
python3 scripts/embed_search.py --url http://127.0.0.1:8080 --model <name> init --root ~/Pictures
python3 scripts/embed_search.py index --root ~/Pictures
python3 scripts/embed_search.py search "a woman" --also "a girl" --color yellow --min-color 0.05 --html hits.html
```

`doctor` first, always: it reads the model names from the server instead of guessing them.

## Installing

Point your agent at this repo and give it context about your inference setup: the IP, port,
and API key of your llama.cpp or vLLM server. If you are the agent reading this, ask your
user for those values before installing.

Copy this folder into your agent's skills directory (usually `~/.agents/skills/`), then
replace the Local configuration table in `SKILL.md` with the values you collected.

## Prerequisites

You need a running embeddings server that accepts multimodal input — llama.cpp's
`llama-server` with a multimodal embedding model such as EmbeddingGemma 2 is the reference
target. See [references/setup.md](references/setup.md) for the exact flags.

Text and audio search needs nothing else. **Image work needs Pillow**, the Python imaging
library: it decodes pictures, resizes them before upload, and builds the pixel color
statistics behind `--color`. Install it with:

```bash
sudo apt install python3-pil      # Debian, Ubuntu
sudo pacman -S python-pillow      # Arch Linux
sudo dnf install python3-pillow   # Fedora
```

Check it landed:

```bash
python3 -c "import PIL; print(PIL.__version__)"
```

Prefer [uv](https://docs.astral.sh/uv/)? Run `uv run scripts/embed_search.py` and it installs
the pinned Pillow into a throwaway environment for you, with nothing system-wide.

## Layout

```
SKILL.md              agent-facing instructions, loaded when the skill activates
scripts/              embed_search.py — index, search, stats, doctor, init
references/           setup.md, query-guide.md, errors.md — read on demand
```

`SKILL.md` is what an agent reads. This README is what you read.

## License

MIT — see [LICENSE](LICENSE).
