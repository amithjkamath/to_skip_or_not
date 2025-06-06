#!/bin/bash

python ./toskipornot/train_2d_unet.py
python ./toskipornot/train_2d_vnet.py
python ./toskipornot/train_2d_attunet.py
python ./toskipornot/train_2d_unetplusplus.py
python ./toskipornot/train_2d_noskipunet.py
python ./toskipornot/train_2d_noskipvnet.py