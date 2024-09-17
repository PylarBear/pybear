# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import os
import io
import glob
import base64



# THIS ISNT WORKING ###########################################################
# ic = skimage.io.ImageCollection(os.path.join(f'{basepath}', f'*.jpeg'))

# filename = os.path.join("\\users","bill","desktop",f"Z34_{idx+1}.txt")
# with open(filename, mode='wb') as f:
#     f.write(base64.b64encode(ic[idx]))  <===== THIS ISNT GIVING THE CORRECT ENCODING
#     f.close

# THIS WORKS ##################################################################
# with io.open(os.path.join(f'{basepath}', f'2005_10_27_13_52_00.jpg'), 'rb') as f:
#     img_data = f.read()

# filename = os.path.join("\\users","bill","desktop",f"Z34_{idx+1}.txt")
# with open(filename, mode='wb') as f:
#     f.write(base64.b64encode(img_data))
#     f.close()

# THE DIFFERENCE IS USING io.open() AND .read() TO OPEN IMAGE AS OPPOSED TO
# skimage.io.ImageCollection OR skimage.io.imread HAVE NO IDEA WHY THIS IS
# HAPPENING



# CODE FOR CREATING BASE64 EMBEDS

if os.name == 'posix':
    basepath = os.path.join('/home', 'bear','Desktop')
    dumppath = os.path.join('/home', 'bear','Desktop')
elif os.name == 'nt':
    basepath = os.path.join('\\Users', 'Bill','Documents','BEAR_DOCUMENTS','PICTURES','MONTE_CARLO')
    dumppath =  dumppath = os.path.join("\\users","bill","desktop")

FILES = glob.glob(os.path.join(basepath, "*.jpg"))

for idx, filename in enumerate(FILES):

    print(f'Reading file {idx+1}... ', end='')
    with io.open(filename, 'rb') as f:
        img_data = f.read()
        f.close()

    print(f'Writing base64 encoding to dump file... ', end='')
    try: os.remove(dumppath)
    except: pass
    with open(os.path.join(dumppath, f"Z34_{idx+1}.txt"), mode='wb') as f:
        f.write(base64.b64encode(img_data))
        f.close()
    print(f'Done.')



