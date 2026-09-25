"""Fetches the SO101 MJCF and meshes from the pinned so101-nexus wheel.

so101-nexus needs Python 3.12 and Isaac Sim ships 3.10, so the wheel is only
downloaded and unzipped, never installed. The Isaac MJCF importer accepts a
single top-level <default> block, and the SO101 file has two, so they are
merged into so101_merged.xml next to the original.
"""

import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET   # trusted input: a pinned wheel we downloaded ourselves
from pathlib import Path

WHEEL_SPEC = "so101-nexus==0.5.1"
ASSET_DIR = Path(__file__).resolve().parent / "assets"
SO101_DIR = ASSET_DIR / "SO101"
MERGED_XML = SO101_DIR / "so101_merged.xml"
WHEEL_PREFIX = "so101_nexus/assets/SO101/"


def merge_top_level_defaults(src, dst):
    tree = ET.parse(src)
    root = tree.getroot()
    defaults = root.findall("default")
    first = defaults[0]
    for extra in defaults[1:]:
        for child in list(extra):
            first.append(child)
        root.remove(extra)
    tree.write(dst)


def download_wheel():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
           "--python-version", "3.12", "-d", str(ASSET_DIR), WHEEL_SPEC]
    subprocess.run(cmd, check=True)
    wheels = list(ASSET_DIR.glob("so101_nexus-*.whl"))
    return wheels[0]


def extract_so101(wheel):
    scratch = ASSET_DIR / "_wheel"
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.startswith(WHEEL_PREFIX):
                z.extract(name, scratch)
    shutil.move(str(scratch / WHEEL_PREFIX), str(SO101_DIR))
    shutil.rmtree(scratch)


def gripperframe_offset(xml_path):
    """(pos, quat wxyz) of the gripperframe site in the gripper body frame, as MuJoCo defines it."""
    root = ET.parse(xml_path).getroot()
    for body in root.iter("body"):
        if body.get("name") != "gripper":
            continue
        for site in body.findall("site"):
            if site.get("name") == "gripperframe":
                pos = [float(v) for v in site.get("pos", "0 0 0").split()]
                quat = [float(v) for v in site.get("quat", "1 0 0 0").split()]
                return pos, quat
    raise ValueError("gripperframe site not found in " + str(xml_path))


def ensure_so101_mjcf():
    """Returns the path of the importer-ready MJCF, fetching assets on first use."""
    if MERGED_XML.exists():
        return str(MERGED_XML)
    if not SO101_DIR.exists():
        wheel = download_wheel()
        extract_so101(wheel)
    merge_top_level_defaults(SO101_DIR / "so101_new_calib.xml", MERGED_XML)
    return str(MERGED_XML)


if __name__ == "__main__":
    print(ensure_so101_mjcf())
