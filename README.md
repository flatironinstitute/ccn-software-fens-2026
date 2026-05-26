# CCN Software Workshop at FENS, June 2026

Materials for CCN software software at the FENS Forum, June 2026.

We have a slack channel for communicating with attendees, if you haven't received an invitation, please send us a note!

> [!NOTE]
> The rest of this README is for contributors to the workshop.

## Building the site locally

To build the site locally, clone this repo and install it in a fresh python 3.12 environment (`pip install -e .`). Then run `make -C docs html O="-T"` and open `docs/build/html/index.html` in your browser.

When building the docs, you can use the `RUN_NB` environment variable to limit the notebooks that are executed. To do so, set it to a comma separated list of strings: any notebooks whose name is a substring of one of the strings will run. For example, `RUN_NB=current,head` will run `group_projects/01_head_direction.md` and `live_coding/02_current_injection.md`.

## strip_text.py

This script creates two copies of each file found at `docs/source/full/*/*md`, the copies are placed at `docs/source/users/*/*.md` and `docs/source/presenters/*/*.md`. Neither of these copies are run; the presenters version is intended as a reference for presenters, while the users version is what users will start with.

For this to work:
- The title should be on a line by itself, use `#` (e.g., `# My awesome title`) and be the first such line (so no comments above it).
- All headers must be markdown-style (using `#`), rather than using `------` underneath them.
- You may need to place blank newlines before/after any `div` opening or closing. I think if you don't place newlines after the `div` opening, it will consider everything after it part of a markdown block (which is probably not what you want if it's a `{code-cell}`).
- divs should not cover any headers: if a header happens in the middle of a div, you'll have to close right before hand and open a new one right after (this is because all headers are rendered in all versions).

Full notebook:
- Will not render any markdown wrapped in a div with `class='render-user'` or `class='render-presenter'` (but will render those wrapped in `class='render-all'`)
- Will not render or run any code wrapped in a div at all! Thus, for code that you want in all notebooks, add `:tag: [render-all]`, but for code that you only want in the user / presenter notebook, wrap it in a div with `class='render-user'` / `class='render-presenter'`. 
- Similarly, wrapping colon-fence blocks (which use `:::`, e.g., admonitions) are messed up when you wrap them in a `div`. But they have a `:class:` attribute themselves, so just add the appropriate `render` class there. See the "Download" admonition at the top of each notebook for an example.

Presenters version preserves:
- All markdown headers.
- All code blocks.
- Only colon-fence blocks (e.g., admonitions) that have the class `render-presenter` or `render-all`
- Only markdown wrapped in a `<div class='render-presenter'>` or `<div class='render-all'>`.

Users version preserves:
- All markdown headers.
- Only code blocks with `:tag: [render-all]` *OR* wrapped in a `<div class='render-user'>`. For code blocks in render-user divs, you should probably also add the `skip-execution` tag
- Only colon-fence blocks (e.g., admonitions) that have the class `render-user` or `render-all`
- Only markdown wrapped in a `<div class='render-user>` or `<div class='render-all'>`.

## Participants

We build all of the notebooks as a sphinx site, which we can browse. `index.md` is the main way in which participants will view the information present in this repo. They are instructed to clone this repo and run `scripts/setup.py`, which will download all the data, run `scripts/strip_text.py`, and convert the resulting user notebooks to `ipynb` files, placing them in the `notebooks/` directory. This is what users will see on their screen as they work on their notebook.

This means that you should check what these notebooks look like to make sure everything renders correctly. In particular, intersphinx **does not** work here (though it does work on the built website). Thus, you should instead use explicit links to the relevant documentation, where necessary. 

For internal cross-references (i.e., those that refer to other notebooks in this site), you can use myst cross-references as normal. When writing anchors for these references, they should end with `-full`, (i.e., `(fit-glm-full)=`). When *referencing* these anchors, that final tag should correspond to which version you want to link to (`-users`, `-presenters` or `-full`; we replace `-full` with the appropriate tag in `strip_text.py`). We will also make sure these point correctly in the user notebooks.
    
## binder

See [nemos Feb 2024 workshop](https://github.com/flatironinstitute/nemos-workshop-feb-2024) for details on how to set up the Binder

For TAs / instructors: [group project link](https://flatironinstitute.github.io/neurorse-workshops/workshops/fens-2026/branch/main/full/group_projects/)
