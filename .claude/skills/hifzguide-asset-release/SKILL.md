---
name: hifzguide-asset-release
description: Create a new HifzGuide asset release on GitHub with correct manifest checksums. Use when quran.db, models, fonts, or other assets have been updated and need to be published to a new release so client apps can download them.
---

# HifzGuide Asset Release Skill

This skill creates a new versioned release on this (`sysofwan/HifzGuide`) repo containing the
downloadable assets that client apps fetch on first launch.

**Why this matters:** The `manifest.json` in each release contains SHA256 checksums for every
asset file. If a checksum doesn't match the actual file, clients will fail to download and get
stuck on old assets.

## Prerequisites

- `gh` CLI authenticated with access to `sysofwan/HifzGuide`
- Run from a clone of this repo

## Background: Asset Generation

All assets are generated in this repo. The key generators:

| Asset | Generator | Notes |
|-------|-----------|-------|
| `quran.db` | `tools/generate_quran_db.py` | Builds SQLite DB from `data/` sources |
| `models.zip` | `tools/compile_models.sh` | Compiles CoreML `.mlpackage` → `.mlmodelc`, then zips |
| `fonts.zip` | Manual | QCF2 font TTFs, zipped |
| `mel_filters.bin` | Exported binary | From ML pipeline |
| `window.bin` | Exported binary | From ML pipeline |
| `manifest.json` | Generated at release time | **Must be generated from the actual files** |

To regenerate assets:
```bash
cd tools && pip install -r requirements.txt
python generate_quran_db.py        # → quran.db
python convert_to_coreml.py        # → .mlpackage models
python palettize_chunks.py         # → quantized models
bash compile_models.sh             # → models.zip
```

## Step 1: Determine the New Version

```bash
gh release list --repo sysofwan/HifzGuide --limit 3
```

Increment the patch version (e.g., `v1.0.4` → `v1.0.5`). Ask the user if unsure.

## Step 2: Identify Changed Assets

Ask the user which assets changed. Only changed assets need to be rebuilt. Unchanged assets are
downloaded from the previous release.

## Step 3: Collect All Asset Files

Stage all files into `/tmp/` with their **exact release names**. This is critical —
`gh release upload` uses the local filename.

### For changed assets
Copy from this repo's build output:
```bash
# Example: quran.db was regenerated
cp tools/quran.db /tmp/quran.db
```

### For unchanged assets
Download from the previous release:
```bash
PREV_TAG=v1.0.4  # adjust to actual previous tag
for f in models.zip fonts.zip quran.db mel_filters.bin window.bin; do
  if [ ! -f "/tmp/$f" ]; then  # skip files already staged
    echo "Downloading $f from $PREV_TAG..."
    curl -sL -o "/tmp/$f" "https://github.com/sysofwan/HifzGuide/releases/download/$PREV_TAG/$f"
  fi
done
```

## Step 4: Generate manifest.json

**CRITICAL:** Always generate the manifest from the ACTUAL files in `/tmp/`. Never copy a
manifest from a previous release.

```bash
NEW_TAG=v1.0.5  # adjust to new version

python3 -c "
import hashlib, json, os

assets = []
for filename in ['models.zip', 'fonts.zip', 'quran.db', 'mel_filters.bin', 'window.bin']:
    path = f'/tmp/{filename}'
    size = os.path.getsize(path)
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assets.append({'filename': filename, 'sha256': sha, 'size': size})
    print(f'  {filename}: sha256={sha} size={size}')

manifest = {'version': '$NEW_TAG', 'assets': assets}
with open('/tmp/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=4)
    f.write('\n')
print()
print('Manifest written to /tmp/manifest.json')
"
```

## Step 5: Verify Checksums

Before uploading, verify every asset's checksum in the manifest matches its file:

```bash
python3 -c "
import hashlib, json

manifest = json.load(open('/tmp/manifest.json'))
all_ok = True
for asset in manifest['assets']:
    path = f'/tmp/{asset[\"filename\"]}'
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    match = '✓' if actual == asset['sha256'] else '✗ MISMATCH'
    if actual != asset['sha256']: all_ok = False
    print(f'  {asset[\"filename\"]}: {match}')

print()
print('All checksums valid!' if all_ok else 'ERROR: Checksum mismatch detected!')
"
```

**Do NOT proceed if any checksum fails.** Re-generate the manifest from Step 4.

## Step 6: Create the GitHub Release

**IMPORTANT:** `gh release create/upload` uses the **local filename** as the asset name. The
`#name` rename syntax does NOT work reliably. That's why Step 3 stages files with their exact
release names.

```bash
NEW_TAG=v1.0.5
TITLE="v1.0.5 — <brief description of what changed>"
NOTES="<description of changes>"

gh release create "$NEW_TAG" \
  --repo sysofwan/HifzGuide \
  --title "$TITLE" \
  --notes "$NOTES" \
  /tmp/manifest.json /tmp/quran.db /tmp/mel_filters.bin /tmp/window.bin
```

## Step 7: Upload Large Assets

The small files were included in `gh release create`. Upload the large ones separately:

```bash
gh release upload "$NEW_TAG" /tmp/fonts.zip --repo sysofwan/HifzGuide
gh release upload "$NEW_TAG" /tmp/models.zip --repo sysofwan/HifzGuide
```

Verify all asset names are correct:
```bash
gh release view "$NEW_TAG" --repo sysofwan/HifzGuide
```

Expected names: `manifest.json`, `quran.db`, `models.zip`, `fonts.zip`, `mel_filters.bin`,
`window.bin`. If any name is wrong (e.g., has a prefix like `release_`), delete the release and
redo from Step 3.

## Step 8: Notify Downstream Consumers

Client apps pin the release tag and the expected `manifest.json` SHA256. After publishing, the
consumer repo(s) must be updated with the new tag and manifest hash. Compute and share the hash:

```bash
shasum -a 256 /tmp/manifest.json
```

This repo owns asset publishing only — the client-side update (base URL + expected manifest hash)
happens in the consuming app's repo.

## Common Pitfalls

| Pitfall | Prevention |
|---------|-----------|
| Manifest has wrong checksum | **Always** generate manifest from the actual files (Step 4), never copy from a previous release |
| Manifest has wrong file size | Same — `os.path.getsize()` reads the real size |
| Asset uploaded with wrong filename | `gh` uses the local filename. Stage files with exact names in Step 3 |
| Asset regenerated after manifest was created | Re-run Steps 4–5 whenever any file changes |
| Consumer not updated | Share the new tag + manifest hash (Step 8) so clients detect the release |
