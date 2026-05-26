#!/usr/bin/env python3
"""Convert Visual Coding NWB to NPZ

The NWB file, even for a single subject/session, is 2.6GB. This script converts that NWB
file to a Folder of NPZs, which we can save and load with pynapple, containing only the
relevant information for the visual_coding.md notebook.

This file is included for reference only -- it **does not** need to be run.

"""

import workshop_utils
import nemos as nmo
import pynapple as nap
from pathlib import Path
import zipfile

output_dir = Path("data/visual_coding_data")
output_dir.mkdir(exist_ok=True)

dandiset_id = "000021"
dandi_filepath = "sub-726298249/sub-726298249_ses-754829445.nwb"
nwbfile = nmo.fetch.download_dandi_data(dandiset_id, dandi_filepath)
data = nap.NWBFile(nwbfile.read())

# grab the spiking data
units = data["units"]

# map from electrodes to brain area
channel_probes = {}
for elec in data.nwb.electrodes:
    channel_id = elec.index[0]
    location = elec["location"].values[0]
    channel_probes[channel_id] = location

# Add a new column to include location in our spikes TsGroup
units.brain_area = [channel_probes[int(ch_id)] for ch_id in units.peak_channel_id]

# drop unnecessary metadata
units.restrict_info(["rate", "quality", "brain_area"])

flashes = data["flashes_presentations"]
flashes.restrict_info(["color"])

flashes.save(output_dir / "flashes.npz")

units = units[units.brain_area == "VISp"]
units.save(output_dir / "units.npz")

# Then to load from folder:
# folder = nap.load_folder(output_dir)
# units = folder["units"]
# flashes = folder["flashes"]

with zipfile.ZipFile(output_dir.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
    for f in output_dir.iterdir():
        zf.write(f, f.name)

# Then to load from zip:
# with zipfile.ZipFile(output_dir.with_suffix(".zip")) as zf:
#     zf.extractall(output_dir)
# folder = nap.load_folder(output_dir)
# units = folder["units"]
# flashes = folder["flashes"]
