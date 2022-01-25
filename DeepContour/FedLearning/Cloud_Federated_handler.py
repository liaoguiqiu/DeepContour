# this script will seperated from the local machine, 
# Enventually it will only be executed with encrpted cloud computation
from Cloud_API import Cloud_API
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from working_dir_root import config_root,Output_root
import json as JSON
import numpy as np
import os
class Cloud_Federated_Handler(object):
    def __init__(self):
        self.credential_dir = config_root + "client_secrets.json"
        self.federated_dir  = config_root + "CEnet_fed/"

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()

        self.drive = GoogleDrive(gauth)
        self.json_dir = self.federated_dir + 'telecom/federated_learning_status.json'  # load json file for drive/cloud and local information
        with open(self.json_dir) as f_dir:
            self.fed_json_data = JSON.load(f_dir)
        # gdrive_id = '1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o'

        # self.fed_drive_id = json_data['federated cloud id']
        # self.upload_file_list = [self.out_dir + 'cGAND.pth',
        #                          self.out_dir + 'cGANG.pth',
        #                          self.json_dir]
        # self.out_dir = Output_root + "CEnet_trained/"

   # add new step of all signals

    def check_jsons_from_works(self):
        num_worker =int( self.fed_json_data["machine number"])
        for i in np.arange(num_worker):
            worker_id = i +1
            this_gdrive_id = self.fed_json_data["worker cloud id"][str(i+1)]
            this_file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
            for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                if file['fileExtension'] =='json':
                    print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j, len(this_file_list)))
                    file.GetContentFile(self.federated_dir + "telecom/worker" + str(worker_id)+".json")
            pass
        pass
#check pth
        for i in np.arange(num_worker):
            worker_id = i + 1
            this_gdrive_id = self.fed_json_data["worker cloud id"][str(i + 1)]
            this_file_list = self.drive.ListFile(
                {'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
            for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                if file['fileExtension'] == 'pth':
                    _,original_name =  os.path.split(file['title'])
                    print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j, len(this_file_list)))
                    file.GetContentFile(self.federated_dir +  str(worker_id) + original_name)
            pass
        pass
# Path_dir, image_name = os.path.split(this_folder_list[0])
	        
       

if __name__ == '__main__':

    Handle = Cloud_Federated_Handler()
    # self test
    Handle.check_jsons_from_works()