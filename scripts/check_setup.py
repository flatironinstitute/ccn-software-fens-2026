#!/usr/bin/env python3

try:
    from rich import print
except ImportError:
    print("rich not found. Try running pip install rich.")
    print("The following will not be pretty...")
import glob
import pathlib
import re
import sys
import subprocess
import warnings
import os
from packaging.version import Version

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

repo_dir = pathlib.Path(__file__).parent.parent
os.environ["NEMOS_DATA_DIR"] = os.environ.get("NEMOS_DATA_DIR", str(repo_dir / "data"))

errors = 0

python_version = sys.version.split('|')[0]
if '3.12' in python_version:
    print(f":white_check_mark: Python version: {python_version}")
else:
    print(f":x: Python version: {python_version}. Create a new virtual environment.")
    errors += 1

try:
    import nemos
except ModuleNotFoundError:
    errors += 1
    print(":x: Nemos not found. Try running [bold]pip install nemos[/bold]")
else:
    print(f":white_check_mark: Nemos version: {nemos.__version__}")

try:
    import pynapple as nap
except ModuleNotFoundError:
    errors += 1
    print(":x: pynapple not found. Try running [bold]pip install pynapple[/bold]")
else:
    print(f":white_check_mark: pynapple version: {nap.__version__}")

p = subprocess.run(['jupyter', '--version'], capture_output=True)
if p.returncode != 0:
    errors += 1
    print(":x: jupyter not found. Try running [bold]pip install jupyter[/bold]")
else:
    # convert to str from bytestring
    stdout = '\n'.join(p.stdout.decode().split('\n')[1:])
    print(f":white_check_mark: jupyter found with following core packages:\n{stdout}")

try:
    import jupyterlab
except ModuleNotFoundError:
    errors += 1
    print(":x: jupyterlab not found.")
else:
    version = Version(jupyterlab.__version__)
    if version < Version("4.5.0"):
        errors += 1
        print(":x: jupyterlab not recent enough! We need at least 4.5.0.")
    else:
        print(f":white_check_mark: jupyterlab version:\n{version}")

p = subprocess.Popen(['jupyter', 'labextension', 'list'], stderr=subprocess.PIPE)
if os.name == "nt":
    search_cmd = "findstr"
else:
    search_cmd = "grep"
try:
    output = subprocess.check_output([search_cmd, 'myst'], stdin=p.stderr).decode().lower()
    p.wait()
except subprocess.CalledProcessError:
    errors += 1
    print(":x: jupyterlab_myst not found. Try running [bold]pip install jupyterlab_myst[/bold]")
else:
    if 'enabled' in output and 'ok' in output:
        with warnings.catch_warnings():
            # this import may give a deprecation warning about how jupyter handles paths
            warnings.simplefilter("ignore")
            import jupyterlab_myst
            version = Version(jupyterlab_myst.__version__)
        if version < Version("2.6.0"):
            errors += 1
            print(":x: jupyterlab_myst not recent enough! We need at least 2.6.0.")
        else:
            print(f":white_check_mark: jupyterlab_myst version:\n{version}")
    else:
        errors += 1
        print(":x: jupyterlab_myst not set up correctly! Look at the output of `jupyter labextension list` and try running [bold]pip install jupyterlab_myst[/bold]")

repo_dir = pathlib.Path(__file__).parent.parent / 'notebooks'
gallery_dir = pathlib.Path(__file__).parent.parent / 'docs' / 'source' / 'full'
nbs = list(repo_dir.glob('**/*ipynb'))
gallery_scripts = [nb for nb in list(gallery_dir.glob('**/*md'))
                   if 'checkpoint' not in nb.name]
missing_nb = [f.stem for f in gallery_scripts
              if not any([f.stem == nb.stem.replace('-users', '') for nb in nbs])]
# index isn't a notebook, so don't check for it
missing_nb = [f for f in missing_nb if f != "index"]
if len(missing_nb) == 0:
    print(":white_check_mark: All notebooks found")
else:
    errors += 1
    print(f":x: Following notebooks missing: {', '.join(missing_nb)}")
    print("   Did you run [bold]python scripts/setup.py[/bold]?")

try:
    import workshop_utils
except ModuleNotFoundError:
    errors += 1
    print(f":x: workshop utilities not found. Try running [bold]pip install .[/bold] from the github repo.")
else:
    missing_files = []
    duplicate_files = []
    from nemos.fetch.fetch_data import _create_retriever
    retriever = _create_retriever()
    for f in workshop_utils.DOWNLOADABLE_FILES:
        # as far as I could find, retriever doesn't have a "check if file is downloaded"
        # function. (is_available just checks the *url* is available). the data file may
        # live in a subdirectory, so search recursively. escape f so filenames are matched
        # literally rather than being interpreted as glob patterns.
        matches = list(retriever.abspath.rglob(glob.escape(f)))
        if len(matches) == 0:
            missing_files.append(f)
        elif len(matches) > 1:
            duplicate_files.append(f)
    if missing_files:
        errors += 1
        print(f":x: Following data files not downloaded: {', '.join(missing_files)}")
        print("   Did you run [bold]python scripts/setup.py[/bold]?")
    if duplicate_files:
        errors += 1
        print(f":x: Following data files found in more than one location: {', '.join(duplicate_files)}")
    if not missing_files and not duplicate_files:
        print(":white_check_mark: All data files found!")


# the expected check figures are whatever the gallery notebooks save via savefig(...)
# into _static/_check_figs/. scan the markdown sources rather than hardcoding the
# filenames, so the check stays in sync when figures are added or removed.
savefig_re = re.compile(r"""savefig\(\s*["'][^"']*_check_figs/([^"'/]+\.png)["']""")
figure_checks = set()
for md in gallery_dir.glob("**/*md"):
    if "checkpoint" in md.name:
        continue
    figure_checks |= set(savefig_re.findall(md.read_text()))
figure_check_dir = pathlib.Path(__file__).parent.parent / "docs" / "source" / "_static" / "_check_figs"
found_figs = set([f.name for f in figure_check_dir.glob("*png")])
if figure_checks - found_figs:
    errors += 1
    print(":x: Some check figures missing. Did you run [bold]python scripts/setup.py[/bold]?")
    print("Missing figures:")
    for fname in (figure_checks - found_figs):
        print(fname)

else:
    print(":white_check_mark: All check figures found!")


if errors == 0:
    print("\n:tada::tada: Congratulations, setup successful!")
    print("\nPlease run `jupyter lab notebooks/live_coding/02_current_injection-users.ipynb`, ")
    print("and ensure that you can run the first few cells (up until the cell containing ")
    print("`path = workshop_utils.fetch_data(\"allen_478498617.nwb\")`).")
else:
    print(f"\n:worried: [red bold]{errors} Errors found.[/red bold]\n")
    print("Unfortunately, your setup was unsuccessful.")
    print("Try to resolve following the suggestions above.")
    print("If you encountered many installation errors and are *not* using uv, run [bold] pip install -e .[/bold] (note the dot!)")
    print("If you are unable to fix your setup yourself, please come to the setup help in ")
    print("the lobby of the event hotel (Meliá Barcelona Sky, Pere IV 272-286 Barcelona 08005)")
    print("during the afternoon of July 2nd, 2026. ")
    print("Be prepared to show us the output of this command, so we can try and fix your problem as quickly as possible!")
