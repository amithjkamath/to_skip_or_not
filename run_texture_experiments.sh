#!/bin/bash

#python ./toskipornot/models/train_glas_2d_full.py
#python ./toskipornot/models/train_busi_2d_full.py
#python ./toskipornot/models/train_spleen_2d_full.py
#python ./toskipornot/models/train_heart_2d_full.py

#python ./toskipornot/models/train_2d_attunet_celoss.py
#python ./toskipornot/models/train_2d_attunet.py
#python ./toskipornot/models/train_2d_unet_celoss.py
#python ./toskipornot/models/train_2d_unet.py
#python ./toskipornot/models/train_2d_noskipunet_celoss.py
#python ./toskipornot/models/train_2d_noskipunet.py

python ./toskipornot/models/train_2d_noskipvnet.py
python ./toskipornot/models/train_2d_vnet.py
python ./toskipornot/models/train_2d_unetplusplus.py
