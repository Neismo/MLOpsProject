Generating the docs
----------

Use [MkDocs](https://www.mkdocs.org/) with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) to update the documentation. This repository uses
[uv](https://docs.astral.sh/uv/getting-started/installation/) to run commands.

Build locally with:

    uv run mkdocs build -f mkdocs.yaml

Serve locally with:

    uv run mkdocs serve -f mkdocs.yaml
