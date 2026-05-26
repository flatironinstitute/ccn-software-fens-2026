#!/usr/bin/env python3

try:
    import click
    import pynapple
    import nemos
except ImportError:
    raise ImportError(
        "We couldn't find an important package, which likely means"
        " we're not running from the right virtual "
        "environment! Did you forget to activate your virtual env"
        " (using `source` or `conda activate`, depending on how "
        "you set it up)"
    )
import pathlib
import shutil
import subprocess
import re
import os
import warnings

with warnings.catch_warnings():
    # will give a warning about documentation utils
    warnings.simplefilter("ignore")
    import workshop_utils


def extract_anchors(md_text):
    anchors = re.findall(r"^\(.*?\)=", md_text, flags=re.MULTILINE)
    return [a.replace("(", "").replace(")=", "") for a in anchors]


@click.command()
def main():
    repo_dir = pathlib.Path(__file__).parent.parent
    nb_dir = repo_dir / "notebooks"
    scripts_dir = repo_dir / "scripts"
    src_dir = repo_dir / "src"
    print("Checking whether we need to download data...")
    workshop_utils.fetch_all()
    docs_nb_dir_gp = repo_dir / "docs" / "source" / "full" / "group_projects"
    docs_nb_dir = repo_dir / "docs" / "source" / "users"
    print("Preparing notebooks, this may take ~5 minutes...")
    shutil.rmtree(docs_nb_dir / "live_coding", ignore_errors=True)
    shutil.rmtree(docs_nb_dir / "group_projects", ignore_errors=True)
    shutil.rmtree(
        repo_dir / "docs" / "source" / "_static" / "_check_figs", ignore_errors=True
    )
    shutil.rmtree(
        repo_dir / "docs" / "source" / "presenters" / "live_coding", ignore_errors=True
    )
    shutil.rmtree(
        repo_dir / "docs" / "source" / "presenters" / "group_projects",
        ignore_errors=True,
    )
    subprocess.run(["python", repo_dir / "scripts" / "strip_text.py"], cwd=repo_dir)

    for f in docs_nb_dir_gp.glob("*md"):
        if "index.md" in f.name or "03_visual_coding" in f.name:
            continue
        # have jupytext write output to a tmp file that we then delete. if --output is
        # not specified, will write to original file which, for some reason, removes the
        # `no-search: true` in the front matter. if the tmp file is not in the same
        # directory as the actual file, relative paths don't work.
        subprocess.run(
            ["jupytext", "--execute", f.absolute(), "--output", f.with_name("tmp.md")]
        )
    # move outside the loop to avoid potential race conditions with next subprocess
    os.remove(f.with_name("tmp.md"))

    # find the file that each anchor lives in, so that we can update them for notebooks
    # below
    anchors = {}
    for f in docs_nb_dir.glob("**/*md"):
        name = f'{f.parent.name}/{f.name.replace(".md", ".ipynb")}'
        for anchor in extract_anchors(f.read_text()):
            anchors[anchor] = name

    for f in docs_nb_dir.glob("**/*md"):
        if "index.md" in f.name:
            continue
        output_f = (nb_dir / f.parent.name / f.name.replace("md", "ipynb")).absolute()
        output_f.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["jupytext", f.absolute(), "-o", output_f, "--from", "myst"], cwd=repo_dir
        )
        nb_contents = re.sub(
            r"../../_static/", r"../../docs/source/_static/", output_f.read_text()
        )
        # update xref anchors from the version that myst wants to one that jupyterlab will
        # recognize (which requires including the file path explicitly)
        for anchor, path in anchors.items():
            tgt_anchor = anchor.replace("-full", "-users").replace(
                "-presenters", "-users"
            )
            src_anchor = "-".join(anchor.split("-")[:-1])
            nb_contents = re.sub(
                rf"]\({src_anchor}-[a-z]+\)", f"](../{path}#{tgt_anchor})", nb_contents
            )
        output_f.write_text(nb_contents)


if __name__ == "__main__":
    main()
