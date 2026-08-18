#!/bin/bash

set -e

VERSION="1.0.0"
APP="morphi/dist/Morphi.app"
PKGROOT="pkgroot"
OUTPUT="Morphi-${VERSION}.pkg"

echo "Building Morphi installer..."

rm -rf "$PKGROOT"
rm -f "$OUTPUT"

mkdir -p "$PKGROOT/Applications"
mkdir -p "$PKGROOT/usr/local/bin"

# Install application
ditto "$APP" "$PKGROOT/Applications/Morphi.app"

# Install command-line launcher
cat > "$PKGROOT/usr/local/bin/morphi" <<'EOF'
#!/bin/sh
open "/Applications/Morphi.app"
EOF

chmod 755 "$PKGROOT/usr/local/bin/morphi"

# Build installer
pkgbuild \
  --root "$PKGROOT" \
  --identifier io.github.JinLab-Westlake.Morphi \
  --version "$VERSION" \
  --install-location / \
  "$OUTPUT"

echo "Created: $OUTPUT"