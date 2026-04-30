##This script has the combine function for calling in files, and the all_amp function
from SSINS import INS
from SSINS import Catalog_Plot as cp
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from custom_funcs import chan_select, chan_avg, DPSS_fit
from pyuvdata import UVFlag


#Code for combining files
def combine(obsid, tar, search_strings, search_directory, save_directory):
    
    '''
    The all_amp function. Utility code for collecting raw INS amplitudes.
    This is what I used to do retrievals on CEDAR

    Args:
        obsid               -- pointings yaml file
        tar                 -- target file
        search_strings      -- set of search strings corresponding to file name.
        search_directory    -- which directory we're searching in
        save_directory      -- which directory we're saving in
    '''
    
    #Loading pointings dictionary.
    with open(obsid, 'r') as file:
        pointings = yaml.safe_load(file)

    #print('Pointings loaded.')
    
    #Searching for the selected obsids and corresponding pointings.
    keys = [
        key for key, val in pointings.items()
        if val == tar
        and any(
            s.lower() in key.lower() 
            for s in search_strings
        )
    ]

    #print('Accessed obsid. Retrieved', len(keys), 'files.')

    all_files = list(Path(search_directory).glob('*'))
    
    #Collecting files in the directory
    data_files = [
        f for f in tqdm(all_files)
        if (f.is_file() 
            and "SSINS_data" in f.name
            and any(key.lower() in f.name.lower() for key in keys))
    ]
    mask_files = [
        f for f in tqdm(all_files)
        if (f.is_file() 
            and "SSINS_mask" in f.name
            and any(key.lower() in f.name.lower() for key in keys))
    ]

    data_files.sort()
    mask_files.sort()

    if len(data_files) == 0:
        raise ValueError("No files collected. Please check that you have entered the correct pointing and search string.")
    
    if len(data_files) - len(mask_files) != 0:
        raise ValueError("Number of data files and mask files do not correspond.")
    
    #print('Proceeding to combine files.')

    #Combining selected files into one large file, and saving on local CEDAR directory.
    combined = INS(data_files[0], mask_file=mask_files[0], telescope_name='mwa')
    for i in tqdm(range(1, len(data_files))):
        current = INS(data_files[i], mask_file=mask_files[i], telescope_name='mwa')
        combined += current

    #print('Files combined successfully.')
    combined.write(save_directory, output_type='data', clobber=True)
    combined.write(save_directory, output_type='mask', clobber=True)

    #print('Combined file saved.')


##File naming -- commented-out files were behaving weirdly when retrieving so ignored them
clean_files_p3 = [
    #['109181', '109182']
    ['108965', '108966'],
    ['108974', '108975'],
    ['108991', '108992'],
    ['109000', '109001'],
    ['109009', '109010'],
    ['109017', '109018'],
    ['109034', '109035'],
    ['109043', '109044'],
    ['109060', '109061'],
    ['109069', '109070'],
    ['109077', '109078'],
    ['109086', '109087'],
    ['109095', '109096'],
    ['109103', '109104'],
    ['109112', '109113'],
    ['109189', '109190'],
    ['109198', '109199'],
    ['109233', '109234'],
    ['109241', '109242'],
    ['109250', '109251'],
    ['109258', '109259'],
    ['109267', '109268'],
    ['109276', '109277'],
    ['109284', '109285'],
    ['109293', '109294'],
    ['109302', '109303'],
    ['109327', '109328'],
    ['109336', '109337'],
    ['109345', '109346'],
    ['109353', '109354'],
    ['109362', '109363'],
    ['109371', '109372'],
    ['109379', '109380'],
    ['109388', '109389'],
    ['109396', '109397'],
    ['109405', '109406'],
    #['109414', '109415']
    ['109422', '109423'],
    ['109431', '109432'],
    ['109439', '109440'],
    ['109448', '109449'],
    ['109474', '109475']
]
clean_files_p1 = [
    #['109181', '109182']
    ['108965', '108966'],
    ['108974', '108975'],
    ['108991', '108992'],
    ['109000', '109001'],
    #['109009', '109010'],
    ['109017', '109018'],
    ['109034', '109035'],
    ['109043', '109044'],
    ['109060', '109061'],
    ['109069', '109070'],
    ['109077', '109078'],
    #['109086', '109087'],
    ['109095', '109096'],
    ['109103', '109104'],
    ['109112', '109113'],
    ['109189', '109190'],
    ['109198', '109199'],
    ['109233', '109234'],
    ['109241', '109242'],
    ['109250', '109251'],
    ['109258', '109259'],
    ['109267', '109268'],
    ['109276', '109277'],
    ['109284', '109285'],
    ['109293', '109294'],
    ['109302', '109303'],
    ['109327', '109328'],
    ['109336', '109337'],
    ['109345', '109346'],
    ['109353', '109354'],
    ['109362', '109363'],
    ['109371', '109372'],
    ['109379', '109380'],
    ['109388', '109389'],
    #['109396', '109397'],
    ['109405', '109406'],
    ['109414', '109415'],
    ['109422', '109423'],
    ['109431', '109432'],
    ['109439', '109440'],
    ['109448', '109449'],
    ['109474', '109475']
]
clean_files_p0 = [
    #['109181', '109182'],
    #['108965', '108966'],
    ['108974', '108975'],
    ['108991', '108992'],
    ['109000', '109001'],
    #['109009', '109010'],
    ['109017', '109018'],
    ['109034', '109035'],
    #['109043', '109044'],
    ['109060', '109061'],
    ['109069', '109070'],
    ['109077', '109078'],
    #['109086', '109087'],
    ['109095', '109096'],
    ['109103', '109104'],
    ['109112', '109113'],
    ['109189', '109190'],
    ['109198', '109199'],
    ['109233', '109234'],
    ['109241', '109242'],
    ['109250', '109251'],
    ['109258', '109259'],
    ['109267', '109268'],
    ['109276', '109277'],
    ['109284', '109285'],
    ['109293', '109294'],
    ['109302', '109303'],
    ['109327', '109328'],
    ['109336', '109337'],
    ['109345', '109346'],
    ['109353', '109354'],
    ['109362', '109363'],
    ['109371', '109372'],
    ['109379', '109380'],
    ['109388', '109389'],
    ['109396', '109397'],
    ['109405', '109406'],
    ['109414', '109415'],
    ['109422', '109423'],
    ['109431', '109432'],
    ['109439', '109440'],
    ['109448', '109449'],
    ['109474', '109475']
]
clean_files_p2 = [
    #['109181', '109182'],
    #['108965', '108966'],
    #['108974', '108975'],
    ['108991', '108992'],
    ['109000', '109001'],
    ['109009', '109010'],
    ['109017', '109018'],
    ['109034', '109035'],
    #['109043', '109044'],
    ['109060', '109061'],
    ['109069', '109070'],
    ['109077', '109078'],
    ['109086', '109087'],
    ['109095', '109096'],
    #['109103', '109104'],
    ['109112', '109113'],
    ['109189', '109190'],
    ['109198', '109199'],
    ['109233', '109234'],
    ['109241', '109242'],
    ['109250', '109251'],
    ['109258', '109259'],
    ['109267', '109268'],
    ['109276', '109277'],
    ['109284', '109285'],
    ['109293', '109294'],
    ['109302', '109303'],
    ['109327', '109328'],
    ['109336', '109337'],
    ['109345', '109346'],
    #['109353', '109354'],
    ['109362', '109363'],
    #['109371', '109372'],
    ['109379', '109380'],
    ['109388', '109389'],
    ['109396', '109397'],
    ['109405', '109406'],
    #['109414', '109415'],
    ['109422', '109423'],
    ['109431', '109432'],
    ['109439', '109440'],
    ['109448', '109449'],
    ['109474', '109475']
]
clean_files_p4 = [
    #['109181', '109182'],
    #['108965', '108966'],
    #['108974', '108975'],
    ['108991', '108992'],
    ['109000', '109001'],
    ['109009', '109010'],
    ['109017', '109018'],
    ['109034', '109035'],
    #['109043', '109044'],
    ['109060', '109061'],
    ['109069', '109070'],
    ['109077', '109078'],
    ['109086', '109087'],
    ['109095', '109096'],
    ['109103', '109104'],
    ['109112', '109113'],
    ['109189', '109190'],
    ['109198', '109199'],
    ['109233', '109234'],
    ['109241', '109242'],
    ['109250', '109251'],
    ['109258', '109259'],
    ['109267', '109268'],
    ['109276', '109277'],
    ['109284', '109285'],
    ['109293', '109294'],
    ['109302', '109303'],
    ['109327', '109328'],
    ['109336', '109337'],
    ['109345', '109346'],
    #['109353', '109354'],
    ['109362', '109363'],
    ['109371', '109372'],
    ['109379', '109380'],
    ['109388', '109389'],
    ['109396', '109397'],
    ['109405', '109406'],
    ['109414', '109415'],
    ['109422', '109423'],
    ['109431', '109432'],
    ['109439', '109440'],
    ['109448', '109449'],
    #['109474', '109475']
]

TV_dict = {
    'TV6': [1.74e8, 1.81e8],
    'TV7': [1.81e8, 1.88e8],
    'TV8': [1.88e8, 1.95e8],
    'TV9': [1.95e8, 2.02e8]
    }


filepath = '/home/andreili/ssins_env/utils/_SSINS_data.h5'
maskpath = '/home/andreili/ssins_env/utils/_SSINS_mask.h5'
prefix = '/home/andreili/ssins_env/utils/'

def all_amp(obsid='/home/andreili/ssins_env/pointing.yaml',
            search_directory='/project/def-acliu/SSINS_tars/tars'):

    '''
    The all_amp function. Utility code for collecting raw INS amplitudes.
    This is what I used to do retrievals on CEDAR

    Args:
        obsid             -- pointings yaml file
        search_directory  -- directory we're searching in
    '''


    search_strings=['1']

    all_amp = []

    for point in [0, 1, 2, 3, 4]:

        print(point)

        tar = point

        with open(obsid, 'r') as file:
            pointings = yaml.safe_load(file)

        #print('Pointings loaded.')
        
        #Searching for the selected obsids and corresponding pointings.
        keys = [
            key for key, val in pointings.items()
            if val == tar
            and any(
                s.lower() in key.lower() 
                for s in search_strings
            )
            and '109181' not in key.lower()
            and '109182' not in key.lower()
        ]

        #print('Accessed obsid. Retrieved', len(keys), 'files.')

        all_files = list(Path(search_directory).glob('*'))
        
        #Collecting files in the directory, and dividing them into 
        data_files = [
            f for f in tqdm(all_files)
            if (f.is_file() 
                and "SSINS_data" in f.name
                and any(key.lower() in f.name.lower() for key in keys))
        ]

        data_files.sort()

        if len(data_files) == 0:
            raise ValueError("No files collected. Please check that you have entered the correct pointing and search string.")
        
        #print('Proceeding to combine files.')

        point_amp = []

        

        for key in TV_dict.keys():
            
            subband_amp = []
            print(key)

            for i in tqdm(range(1, len(data_files))):
                
                ins = INS(data_files[i], telescope_name='mwa')

                ins_subband = chan_select(ins, key, TV_dict)[0]
                amp = chan_avg(ins_subband)[1]

                subband_amp = np.concatenate((subband_amp, amp))
        
            point_amp.append(subband_amp)
        
        all_amp.append(point_amp)
    
    return all_amp


            
                
                
                