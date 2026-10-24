#!/bin/sh
# This machine's run of the matrix, most important first: the two pairs, the check of Fizgig's own
# figure, then the compiled round. (run.sh would do them all in matrix order.)
set -e
cd "$(dirname "$0")"
B="uv run --no-project --with pyyaml --with pillow python bench.py"
$B prepare
for v in zt-w8a8 zt-weightonly aitk-weightonly fizgig-w8a8 \
         zt-w8a8-512-r8 fizgig-w8a8-512-r8 fizgig-w8a8-512-r8-compiled \
         zt-w8a8-compiled zt-weightonly-compiled aitk-weightonly-compiled fizgig-w8a8-compiled; do
  $B run $v
done
$B results
echo MATRIX DONE
