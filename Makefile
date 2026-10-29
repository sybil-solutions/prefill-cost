PY ?= python3
SOURCES ?= ..

.PHONY: all data site check clean

all: site

# Re-extract the machine-read rows from sibling clones of the source repos.
data:
	$(PY) tools/extract.py --sources $(SOURCES)
	$(PY) tools/merge.py

site: data
	$(PY) tools/render.py

check:
	$(PY) tools/merge.py --check
	$(PY) tools/render.py --check

clean:
	rm -f data/prefill.jsonl data/summary.json
	rm -rf tools/__pycache__
