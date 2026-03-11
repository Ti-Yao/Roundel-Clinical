import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
import cv2
import json
import math
import re
from PIL import Image
from pathlib import Path
from scipy import stats
import os
import pickle
from pathlib import Path
from skimage.measure import label 
import pydicom
import streamlit as st

class Pipeline:
    def __init__(self, dicom_files):
        self.dicom_files  = dicom_files
        self.sax_df = self.get_sax_df() # read dicom headers for each file into a dataframe called sax_df
        self.image = self.get_sax_image()# create the sax image

    def get_sax_df(self):
        '''
        puts all the dicom header information for ALL dicoms into a dataframe
        '''

        sax_df = {}
        dicoms_in_series = self.read_dicom_header(self.dicom_files)
        sax_df.update(dicoms_in_series)
        sax_df = pd.DataFrame.from_dict(sax_df, orient = 'index').reset_index(drop = True) # put dicom info for all images into a dataframe

        sax_df = sax_df[sax_df ['triggertime'].notna()] #remove scans with no triggertimes
        if sax_df.slicelocation.isnull().any():
            main_axis = np.argmax(np.cross(sax_df['orientation'].iloc[0][:3], sax_df['orientation'].iloc[0][3:]))
            sax_df['slicelocation'] = sax_df['position'].apply(lambda x: x[main_axis])
        sax_df = sax_df.sort_values(['slicelocation','triggertime'])
        if self.is_sax_valid(sax_df):
            return sax_df
        else:
            st.error('Not a Valid SAX series')
            st.stop()


    def read_dicom_header(self,dicoms_in_series):
        '''
        read the information we want from the header and assert that the series has to have pixelarray data
        '''
        sax_df = {}
        for dicom_num, dicom in enumerate(dicoms_in_series): # go through dicom in each series
            try: # if dicom doesn't have an associate pixel array (image), ignore dicom
                dicom.seek(0)  # reset stream position
                dcm = pydicom.dcmread(dicom) # read dicom
                try:
                    image = dcm.pixel_array
                    image_exists = True
                except:
                    image_exists = False                
                if image.ndim == 3: # ignore dicom if 3d
                    image_exists = False
                try:
                    if dcm.MRAcquisitionType == '3D': # ignore dicom if 3d
                        image_exists = False
                except:
                    pass
            except Exception as e:
                image_exists = False

            if image_exists: # if image exists and is not 3d read all other information
                sax_df[dicom_num] = {}
                sax_df[dicom_num]['image'] = dcm.pixel_array
                sax_df[dicom_num]['uid'] = dcm.SOPInstanceUID

                # have to use try and excepts, if the dicom doesn't the information stored use nan
                try:
                    sax_df[dicom_num]['slicelocation'] = [round(val,3) for val in dcm.ImagePositionPatient][2]
                except:
                    sax_df[dicom_num]['slicelocation'] = np.nan
            
                try:
                    sax_df[dicom_num]['thickness'] = round(dcm.SpacingBetweenSlices,3)
                except:
                    try:
                        sax_df[dicom_num]['thickness'] = round(dcm.SliceThickness,3)
                    except:
                        sax_df[dicom_num]['thickness'] = np.nan
                try:
                    sax_df[dicom_num]['seriesnumber'] = dcm.SeriesNumber
                except:
                    sax_df[dicom_num]['seriesnumber'] = np.nan
                try:
                    sax_df[dicom_num]['triggertime'] = round(dcm.TriggerTime)#int(np.ceil(dcm.TriggerTime / 5) * 5)
                except:
                    sax_df[dicom_num]['triggertime'] = np.nan
                try:
                    sax_df[dicom_num]['N_timesteps'] = int(dcm.CardiacNumberOfImages)
                except:
                    sax_df[dicom_num]['N_timesteps'] = np.nan
                try:
                    sax_df[dicom_num]['orientation'] = [round(val,3) for val in dcm.ImageOrientationPatient]
                except:
                    sax_df[dicom_num]['orientation'] = np.nan
                try:
                    sax_df[dicom_num]['position'] = [round(val,3) for val in dcm.ImagePositionPatient]
                except:
                    sax_df[dicom_num]['position'] = np.nan
                try:
                    sax_df[dicom_num]['pixelspacing'] = round(dcm.PixelSpacing[0],3)
                except:
                    sax_df[dicom_num]['pixelspacing'] = np.nan
                # try:
                #     sax_df[dicom_num]['phase'] = dcm[0x0028, 0x1052].value
                # except:
                #     try:
                #         sax_df[dicom_num]['phase'] = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept 
                #     except:
                #         sax_df[dicom_num]['phase'] = 0
        
        return sax_df

    def get_sax_image(self):
        '''
        makes the 4D sax image image[height, width, slice, time]
        '''
        try:
            image_4D = []
            for uni_slice in self.sax_df.slicelocation.unique():
                image_4D.append(np.stack(self.sax_df.loc[self.sax_df['slicelocation'] == uni_slice].image.values, axis =-1))
            image_4D = np.stack(image_4D, axis = -2)
        except:
            self.status = 'Mismatched timesteps'
            raise ValueError('Mismatched timesteps')
        return image_4D
    
    def calc_N_timesteps(self,sax_df):
        '''
        N timesteps is given in the dicom header as number cardiac images, but it's not always there.
        This calculates the number of timesteps there should be in a series by taking the modal value of the 
        number of trigger times for each series.
        '''
        sax_df = sax_df.drop_duplicates(subset = ['slicelocation','triggertime']) # remove any repeated scans
        unique_slices = sax_df.slicelocation.unique()
        possible_N_timesteps = []
        for uni_slice in unique_slices:
            possible_N_timesteps.append(len(sax_df.loc[sax_df['slicelocation'] == uni_slice]))

        N_timesteps = np.min(possible_N_timesteps)
        return int(N_timesteps)
    
    def is_sax_valid(self,sax_df):
        '''
        checks if a stack is valid as a sax by saying that it has the greater than the minimum number of timesteps and slices
        '''
        min_slices = 6
        min_timesteps = 10
        min_images = 60
        N_timesteps = self.calc_N_timesteps(sax_df)

        N_slices =  sax_df.slicelocation.nunique() 

        if N_slices>= min_slices and N_timesteps >= min_timesteps and len(sax_df) >= min_images and len(sax_df) % N_timesteps ==  0:
            sax_valid = True
        else:
            sax_valid = False
        return sax_valid

    def __iter__(self):
        yield self.image
        yield self.sax_df