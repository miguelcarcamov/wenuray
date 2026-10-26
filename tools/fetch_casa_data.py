"""
Fetch the CASA data Wenuray needs into a directory, without the full casarundata.

- The measures (IERS Earth orientation, ephemerides; ~20 MB) come from the measures tarball.
- The observatory positions come from CASA's maintained ``geodetic/Observatories`` table, which only
  ships inside casarundata (~1 GB). The tarball is streamed and only that table is extracted. The
  measures tarball has its own (ASTRON) table, but it lacks telescopes such as the VLA, and CASA
  looks observatories up by name (e.g. tclean frame conversions, the simulator).

Usage: python tools/fetch_casa_data.py <path> [--measures-site URL]
"""
import argparse
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.request

CASARUNDATA_URL = "https://go.nrao.edu/casarundata"
OBSERVATORIES = "geodetic/Observatories"


def fetch_observatories(path):
    import certifi
    from casaconfig import data_available

    version = sorted(data_available())[-1]
    context = ssl.create_default_context(cafile=certifi.where())
    root = urllib.request.urlopen(CASARUNDATA_URL, context=context).url
    url = f"{root.rstrip('/')}/{version}"
    print(f"Extracting {OBSERVATORIES} from {url}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        with urllib.request.urlopen(url, context=context, timeout=400) as stream, \
                tarfile.open(fileobj=stream, mode="r|*") as tar:
            found = None
            for member in tar:
                name = member.name.lstrip("./")
                if f"/{OBSERVATORIES}/" in f"/{name}" or f"/{name}".endswith(f"/{OBSERVATORIES}"):
                    # Keep the path from "geodetic/" on, whatever prefix the tarball uses
                    member.name = name[name.index(OBSERVATORIES):]
                    tar.extract(member, tmp, filter="data")
                    found = True
                elif found:
                    break  # the table's members are contiguous; skip the rest of the stream
        if not found:
            raise RuntimeError(f"{OBSERVATORIES} not found in {url}")
        target = os.path.join(path, OBSERVATORIES)
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(os.path.join(tmp, OBSERVATORIES), target)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--measures-site", default=None)
    args = parser.parse_args()

    from casaconfig import measures_update

    os.makedirs(args.path, exist_ok=True)
    # Without use_astron_obs_table, measures_update leaves geodetic/Observatories alone
    measures_update(args.path, force=True, measures_site=args.measures_site or None)
    fetch_observatories(args.path)


if __name__ == "__main__":
    main()
