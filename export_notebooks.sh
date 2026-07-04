#!/usr/bin/env bash
# Export the completed workshop student notebooks to HTML + PDF in /results/.
#
# Usage:
#   ./export_notebooks.sh                # export to /results (default)
#   OUT_DIR=/some/dir ./export_notebooks.sh
#
# HTML is produced with nbconvert; PDF uses nbconvert's `webpdf` exporter, which
# renders the notebook in headless Chromium (via Playwright) so plots/images look
# exactly like the HTML. The notebooks are exported AS-IS (their saved outputs are
# used — nothing is re-executed).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/results}"

# The completed, student-facing notebooks (relative to the repo root).
NOTEBOOKS=(
  "session1/notebooks/session1_qc_clustering_visualization.ipynb"
  "session2/notebooks/old/session2_literature_cell_types.ipynb"
  "session2/notebooks/session2_webportal_mapping_spatial.ipynb"
)

mkdir -p "$OUT_DIR"

# --- Ensure the PDF engine (Playwright + headless Chromium) is available --------
ensure_pdf_engine() {
  python -c "import playwright" >/dev/null 2>&1 || {
    echo "[setup] installing nbconvert[webpdf] + playwright ..."
    pip install --quiet "nbconvert[webpdf]" playwright
  }
  # Install the Chromium browser + its OS libraries if not already present.
  if ! python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    p.chromium.launch(args=['--no-sandbox']).close()
" >/dev/null 2>&1; then
    echo "[setup] installing Chromium for Playwright ..."
    playwright install chromium
    # install-deps needs root/apt; ignore failure if libs are already present
    playwright install-deps chromium || true
  fi
}

ensure_pdf_engine

# --- Convert -------------------------------------------------------------------
for nb in "${NOTEBOOKS[@]}"; do
  src="$REPO_DIR/$nb"
  base="$(basename "$nb" .ipynb)"
  if [[ ! -f "$src" ]]; then
    echo "[skip] not found: $src"
    continue
  fi
  echo "=== $base ==="
  jupyter nbconvert --to html \
    --output-dir "$OUT_DIR" --output "$base" "$src"
  jupyter nbconvert --to webpdf --disable-chromium-sandbox \
    --output-dir "$OUT_DIR" --output "$base" "$src"
done

echo
echo "Done. Exports in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.html "$OUT_DIR"/*.pdf 2>/dev/null | sed 's/^/  /'
