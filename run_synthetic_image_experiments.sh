#!/bin/bash

python ./toskipornot/train/2d_unet.py
python ./toskipornot/train/2d_vnet.py
python ./toskipornot/train/2d_attunet.py
python ./toskipornot/train/2d_unetplusplus.py
python ./toskipornot/train/2d_noskipunet.py
python ./toskipornot/train/2d_noskipvnet.py