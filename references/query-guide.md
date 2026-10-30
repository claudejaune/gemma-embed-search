# Query guide

## How the query is built

The script prefixes every text query with `task: search result | query: <your text>`
because that is the format these models are trained on; a bare phrase scores worse. You do
not need to add the prefix yourself. Indexed text files get a `title: none | text: <body>`
prefix instead. Media is sent without any prefix.

Scoring is cosine similarity between the query vector and each stored vector. Higher is
better. Typical spread on a photo corpus: 0.30 unrelated, 0.45 vague, 0.60 clearly related,
0.70 strong. The numbers shift with the model and quantization, so calibrate on your own
corpus instead of trusting these values.

## One facet or several

**Synonyms — `--also`.** Different phrasings of the same idea get averaged into a single
query vector, which stops one lucky word from dominating:

```bash
search "a woman" --also "a girl" --also "female portrait"
```

**Independent conditions — `--must`.** Each facet is embedded separately and the item score
becomes the minimum across facets, so a file must satisfy all of them:

```bash
search "a woman" --must "the color yellow" --must "outdoors"
```

Minimum-based AND is strict: one weak facet sinks the score. If a run returns too little,
drop a `--must` and re-check.

**Color — `--color` plus `--min-color`.** The embedding model blurs adjacent hues; asking
for "yellow" in the query text returns orange and gold images. The color filter is computed
from the stored pixel histogram instead, so it is exact and free at query time:

```bash
search "a woman" --color yellow --min-color 0.05
```

`--min-color` is the fraction of pixels in that hue band. Guidance for photos:

| Ratio | Reading |
| --- | --- |
| 0.01 | A small accent: a logo, a flower, a stripe |
| 0.03–0.05 | The color is clearly present |
| 0.10–0.20 | A dominant theme of the frame |
| above 0.30 | Nearly monochrome in that color |

Start at 0.05 and lower it if too little comes back. `--max-color` inverts the test, which
is how you ask for "no red in it". Names accepted: red, orange, yellow, green, cyan, blue,
purple, pink.

Combining a semantic facet with a color is usually the best recipe: the model decides *what
is in the picture*, the histogram decides *what color it is*.

## Narrowing the candidate set

- `--modality image|audio|text` restricts to one kind of file.
- `--offset N` with `--top N` pages through a long list.
- `--index FILE` or `--root DIR` selects which index to search.

## Shortening vectors

`--dims 512`, `256`, or `128` truncates the vectors (Matryoshka training) and renormalizes.
Quality holds well down to 256 dims and the scan gets proportionally cheaper on very large
indexes. Truncate both sides at once, which the script does; never mix dims between the
stored vectors and the query.

## Tuning loop

1. Run with `--top 20 --html review.html` and open the sheet.
2. Wrong subject: rephrase the query, add `--also` synonyms. Art style is ignored by the
   model, so you never need style words.
3. Right subject, wrong color: adjust `--min-color`, or move the color word out of the query
   text and into `--color`.
4. Too few results: relax a `--must`, lower `--min-color`, or raise `--top`.
5. Nothing looks right at all: confirm the index is not empty (`stats`) and that the model
   in `stats` matches the one you are querying with.

## Recipes

```bash
# Portraits with a yellow theme, any art style
search "a woman" --also "a girl" --color yellow --min-color 0.05 --top 40 --html yellow.html

# Landscapes that are mostly blue
search "landscape scenery" --color blue --min-color 0.20

# Screenshots of an error, not photos
search "an error message on a screen" --modality image --top 10

# Sound of rain among recorded clips
search "rain falling" --modality audio

# Find near-duplicates: query with a file's own meaning
search "the same scene as the reference" --top 30
```

The last recipe is approximate. To find near-duplicates properly, embed the reference file
into its own index and search with its vector; the script does not expose a
file-as-query mode, so ask the user before approximating it.
