"""Build the small camera binding against an existing read-only Luisa build."""

from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sysconfig

parser = argparse.ArgumentParser()
parser.add_argument("--luisa-source", type=Path, required=True)
parser.add_argument("--runtime-bin", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).with_name("luisa_extras.cpp").resolve()
runtime = args.runtime_bin.resolve()
commands = json.loads((args.luisa_source / "build/compile_commands.json").read_text())
original = next(c for c in commands if c["file"].endswith("/apps/py_interface.cpp"))
parts = shlex.split(original["command"])
# Preserve the ABI-related defines and includes of the verified existing build.
flags = parts[1 : parts.index("-o")]
flags = [f for f in flags if not f.startswith("-flto") and f != "-fno-fat-lto-objects"]
output = runtime / ("WorldSolverLuisaExtras" + sysconfig.get_config_var("EXT_SUFFIX"))
libs = ["luisa-render-base", "luisa-render-sdl", "luisa-render-util", "lc-runtime", "lc-ast", "lc-core"]
cmd = [
    parts[0],
    *flags,
    "-shared",
    str(source),
    "-o",
    str(output),
    "-L" + str(runtime),
    "-Wl,-rpath,$ORIGIN",
    *["-l" + lib for lib in libs],
]
subprocess.run(cmd, check=True)
print(output)
