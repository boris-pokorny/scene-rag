#!/usr/bin/env sh
# Walk the scene-api with curl and jq. Needs the role running (`pdm run scene-api`) over stores
# the steps have filled. Usage: examples/scene_api_walkthrough.sh [base-url]
# Every curl fails the script on a non-2xx answer (-f), so an empty store stops at the first
# lookup instead of feeding an error body to jq.
set -eu
BASE="${1:-http://127.0.0.1:8080}"
get() { curl -fsS "$BASE$1"; }

echo "# health"
get /health | jq .

echo "# first page of scene sets, with coverage per enricher and fingerprint"
get "/scene-sets?limit=2" | jq '{items: [.items[] | {version, asset_version, segmentation_fingerprint, scene_count, enrichments}], next}'

SET=$(get "/scene-sets?limit=1" | jq -r '.items[0].version // empty')
[ -n "$SET" ] || { echo "no live scene set yet; run the steps first" >&2; exit 1; }
# Pins of the enrichments complete over the set; empty when none is, which pins nothing.
PINS=$(get "/scene-sets/$SET" | jq -r '[.enrichments[] | select(.complete) | "enrichment=\(.enricher):\(.fingerprint)"] | join("&")')
echo "# scene sets complete under the pins: ${PINS:-(none)}"
get "/scene-sets?limit=100${PINS:+&$PINS}" | jq '[.items[].version]'

echo "# scene records of $SET, pinned, page by page"
AFTER=""
while :; do
  PAGE=$(get "/scene-sets/$SET/scenes?limit=2${PINS:+&$PINS}${AFTER:+&after=$AFTER}")
  echo "$PAGE" | jq '.items[] | {version, index: .scene.index, span: [.scene.start_seconds, .scene.end_seconds], entitlements: .asset.entitlements, enrichments: [.enrichments[] | {enricher, fingerprint, segments: (.segments | length)}]}'
  AFTER=$(echo "$PAGE" | jq -r '.next // empty')
  [ -n "$AFTER" ] || break
done

ASSET=$(get "/scene-sets/$SET" | jq -r '.asset_version')
echo "# asset version $ASSET"
get "/asset-versions/$ASSET" | jq .

URL=$(get "/scene-sets/$SET/scenes/0" | jq -r '.scene.keyframes[0].url')
OUT=$(mktemp "${TMPDIR:-/tmp}/keyframe.XXXXXX")
echo "# keyframe bytes from $URL, saved to $OUT"
curl -fsS -D - -o "$OUT" "$BASE$URL" | grep -iE '^(HTTP|content-type|etag|cache-control)'
ls -l "$OUT"
ETAG=$(curl -fsSI "$BASE$URL" | awk 'tolower($1) == "etag:" {print $2}' | tr -d '\r')
echo "# the same keyframe revalidated with If-None-Match: $ETAG"
curl -sS -o /dev/null -w '%{http_code}\n' -H "If-None-Match: $ETAG" "$BASE$URL"
