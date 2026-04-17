# Contributing

## Documentation Quality Gate

Before opening a docs PR, run the checks below from the project root.

### 1. Validate shell scripts parse

```bash
for f in *.sh; do
  bash -n "$f"
done
```

### 2. Validate documented `python -m` modules exist

```bash
rg -No 'python -m (motsart\.[A-Za-z0-9_\.]+)' README.md docs -g '*.md' | \
sed 's/^[^:]*:python -m //' | sort -u | \
while read -r mod; do
  mod_path="$(printf '%s' "$mod" | sed 's/\./\//g')"
  path="src/${mod_path}.py"
  pkg="src/${mod_path}/__init__.py"
  [[ -f "$path" || -f "$pkg" ]] || echo "Missing module: $mod"
done
```

### 3. Run strict MkDocs build

Install docs dependencies if needed:

```bash
pip install -r docs/requirements.txt
```

Build docs in strict mode:

```bash
python3 -m mkdocs build --strict --site-dir /tmp/motsart-docs-site
```
