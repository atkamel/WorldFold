#!/usr/bin/env bash
# Build the SO-101 closed-form IK.
#   bash build.sh            the library Python loads, portable: any x86-64 CPU from ~2009 on (-march=x86-64-v2)
#                            Linux -> lib/linux-x86_64/libso101_ik.so     Windows (MSYS2 g++) -> lib/win-amd64/so101_ik.dll
#                            plus the benchmark / sweep in build/<platform>/bench_so101_ik
#   bash build.sh native     the same, tuned for THIS machine's CPU, all in build/<platform>-native/ (for benchmarks only)
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
ARCH=${1:-portable}
if [ "$ARCH" = native ]; then MARCH="-march=native"; else MARCH="-march=x86-64-v2 -ffp-contract=off"; fi
# -fno-math-errno: sqrt compiles to the CPU instruction (no libm).  No -ffast-math: results stay IEEE and identical
# on every machine and compiler (the Windows and Linux builds give bit-identical answers).
FLAGS="-std=c++17 -O3 $MARCH -fno-math-errno -fno-trapping-math -fno-exceptions -fno-rtti -Wall -Wextra"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PLAT=win-amd64; LIB=so101_ik.dll; LINK="-shared -static"; EXE=.exe ;;
  *) PLAT=linux-x86_64; LIB=libso101_ik.so; LINK="-shared -fPIC -fvisibility=hidden -static-libgcc -static-libstdc++"; EXE= ;;
esac
if [ "$ARCH" = native ]; then OUT=build/$PLAT-native; BIN=$OUT; else OUT=lib/$PLAT; BIN=build/$PLAT; fi
mkdir -p "$OUT" "$BIN"
$CXX $FLAGS $LINK so101_ik_capi.cpp -o "$OUT/$LIB"
$CXX $FLAGS bench_so101_ik.cpp -o "$BIN/bench_so101_ik$EXE"
echo "built $OUT/$LIB and $BIN/bench_so101_ik$EXE ($CXX $MARCH)"
