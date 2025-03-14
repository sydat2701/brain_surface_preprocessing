import os
import shutil
import numpy as np
import glob
from multiprocessing import Pool
import pandas as pd

RAW_DATA_FOLDER = '/code/SiT/ADNI3_RAW/Nifti'
DATA_FOLDER = '/code/Surface-based_Analysis_for_AD/ADNI1/free7_RAS'

SUBJECT_LISTS = os.listdir('/code/Surface-based_Analysis_for_AD/ADNI1/free7_RAS')
SUBJECT_LISTS.remove('fsaverage')

os.environ["SUBJECTS_DIR"] = os.path.abspath(DATA_FOLDER)

def freesurfer(sub):
    # Run recon-all and gtmseg
    mri_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*MPRAGE*.nii')
    os.system(f"recon-all -all -s {sub} -i {mri_file}> /dev/null")
    os.system(f"gtmseg --s {sub} > /dev/null")

def petsurfer(sub, pet_type, use_pvc=False):
    FWHM = 8
    if pet_type == 'amyloid':
        pet_folder = f'{DATA_FOLDER}/{sub}/pet_uniform/AV45'
        pet_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*AV45*.nii') + glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*FBB*.nii')

    elif pet_type == 'tau':
        pet_folder = f'{DATA_FOLDER}/{sub}/pet_uniform/AV1451'
        pet_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*AV1451*.nii')

    elif pet_type == 'fdg':
        pet_folder = f'{DATA_FOLDER}/{sub}/pet_uniform/FDG'
        pet_file = glob.glob(f'{RAW_DATA_FOLDER}/{sub}/*FDG*.nii')
        
    if len(pet_file) == 0:
        print(f"Can't find {pet_type} PET for {sub}.")
        return False
    pet_file = pet_file[0]

    # Create average template.nii.gz
    os.system(f"mri_concat {pet_file} --mean --o {pet_folder}/template.nii.gz")

    # run the rigid (6 DOF) registration
    os.system(f"mri_coreg --s {sub} --mov {pet_folder}/template.nii.gz --reg {pet_folder}/template.reg.lta")

    # Perform PVC
    if use_pvc:
        os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --psf {FWHM} --seg {DATA_FOLDER}/{sub}/mri/gtmseg.mgz --default-seg-merge --auto-mask 1 .01 --mgx .01 --o {pet_folder} --no-rescale") 

        # Mapping to left hemisphere
        os.system(f"mri_vol2surf --mov {pet_folder}/mgx.ctxgm.nii.gz --reg {pet_folder}/aux/bbpet2anat.lta --hemi lh --projfrac 0.5 --o {DATA_FOLDER}/{sub}/surf/lh.{pet_type}.pvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")

        # Mapping to right hemisphere
        os.system(f"mri_vol2surf --mov {pet_folder}/mgx.ctxgm.nii.gz --reg {pet_folder}/aux/bbpet2anat.lta --hemi rh --projfrac 0.5 --o {DATA_FOLDER}/{sub}/surf/rh.{pet_type}.pvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")

    else:
        os.system(f"mri_vol2surf --mov {pet_folder}/PET_T1.nii.gz --regheader {sub} --hemi lh --projfrac 0.5 --o {DATA_FOLDER}/{sub}/surf/lh.{pet_type}.nopvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")

        os.system(f"mri_vol2surf --mov {pet_folder}/PET_T1.nii.gz --regheader {sub} --hemi rh --projfrac 0.5 --o {DATA_FOLDER}/{sub}/surf/rh.{pet_type}.nopvc.fsaverage.nii.gz --cortex --surf white --trgsubject fsaverage")

def process(sub):
    print(f"Processing: {sub}")
    freesurfer(sub)
    # Amyloid
    petsurfer(sub, 'amyloid', True)
    # Tau
    petsurfer(sub, 'tau', True)
    # FDG
    petsurfer(sub, 'fdg', False)
    print(f"Done processing {sub}")
    return True

with Pool(10) as p:
    print(p.map(process, SUBJECT_LISTS))
