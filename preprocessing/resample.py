'''
    This script resample all metrics file to 6th-order icosphere using human connectome workbench
'''

import os
from multiprocessing import Pool

data_folder = '../data_adni1_v2'

subject_list = os.listdir(data_folder)

def process(sub):
    print(sub)
    files = [
            ## Fsaverage template space
             "lh.thickness.roi.fsaverage", "rh.thickness.roi.fsaverage", 
             "lh.curv.roi.fsaverage", "rh.curv.roi.fsaverage",
             "lh.sulc.roi.fsaverage", "rh.sulc.roi.fsaverage",
#              'lh.amyloid.pvc.roi.fsaverage', 'rh.amyloid.pvc.roi.fsaverage',
#              'lh.tau.pvc.roi.fsaverage', 'rh.tau.pvc.roi.fsaverage',
             'lh.fdg.nopvc.roi.fsaverage', 'rh.fdg.nopvc.roi.fsaverage',
             ]

    for f in files:
        input_file = f'{data_folder}/{sub}/{f}.gii'
        output_file = f'{data_folder}/{sub}/{f}.shape.gii'
        
        if 'lh' in f:
            input_surf = '../Icospheres/lh.average.surf.gii'
            if 'fsaverage' not in f:
                input_surf = f'{data_folder}/{sub}/lh.sphere.surf.gii'

            target_surf = '../Icospheres/ico-6.L.surf.gii'
            os.system(f'wb_command -metric-resample {input_file} {input_surf} {target_surf} BARYCENTRIC {output_file}')
        else:
            input_surf = '../Icospheres/rh.average.surf.gii'
            if 'fsaverage' not in f:
                input_surf = f'{data_folder}/{sub}/rh.sphere.surf.gii'

            target_surf = '../Icospheres/ico-6.R.surf.gii'
            os.system(f'wb_command -metric-resample {input_file} {input_surf} {target_surf} BARYCENTRIC {output_file}')
            os.system(f'wb_command -set-structure {output_file} CORTEX_LEFT')
    
for sub in subject_list:
    process(sub)