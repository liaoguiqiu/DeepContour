# this script will seperated from the local machine, 
# Enventually it will only be executed with encrpted cloud computation
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import CE_build3 # the mmodel
from Cloud_API import Cloud_API
import pydrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from working_dir_root import config_root,Output_root
import json as JSON
import numpy as np
import os
class Cloud_Federated_Handler(object):
    def __init__(self):
        Model_creator = CE_build3.CE_creator()  # the  CEnet trainer with CGAN
        #   Use the same arch to create two nets
        # CE_Nets = Model_creator.creat_nets()  # one is for the contour cordinates
        self.credential_dir = config_root + "client_secrets.json"
        self.federated_dir  = config_root + "CEnet_fed/"

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()

        self.drive = GoogleDrive(gauth)
        self.json_root = self.federated_dir + 'telecom/'
        self.fed_json_dir = self.json_root + 'federated_learning_status.json'  # load json file for drive/cloud and local information
        with open(self.fed_json_dir) as f_dir:
            self.fed_json_data = JSON.load(f_dir)
        self.num_worker = int(self.fed_json_data["machine number"])
        self.model_list = [None]*self.num_worker
        self.workerJsonlist = [None]*self.num_worker
        # creat a list of models depending on workers
        for i in np.arange(self.num_worker):
            self.model_list[i] = Model_creator.creat_nets()
        self.average_model =  Model_creator.creat_nets()
        self.update_flag  = self.fed_json_data['federated update']
        self.gdrive_id = self.fed_json_data['federated cloud id']
        self.upload_model_list = [self.federated_dir + 'cGAND.pth',
                                  self.federated_dir + 'cGANG.pth']
        self.upload_json_list = [self.fed_json_dir]
        # self.fed_drive_id = json_data['federated cloud id']
        # self.upload_file_list = [self.out_dir + 'cGAND.pth',
        #                          self.out_dir + 'cGANG.pth',
        #                          self.json_dir]
        # self.out_dir = Output_root + "CEnet_trained/"

   # add new step of all signals
    def load_fed_json(self):
        with open(self.fed_json_dir) as f_dir:
            self.fed_json_data = JSON.load(f_dir)
            # self.ready_worker_cnt =  self.fed_json_data['ready machine number']
        self.update_flag  = self.fed_json_data['federated update']
    def write_fed_json(self):
        with open(self.fed_json_dir, "w") as jsonFile:
            JSON.dump(self.fed_json_data, jsonFile)
        print("fed json status updated")
    def fed_average(self):
        num_worker = int(self.fed_json_data["machine number"])

        # load weight from all dowaloaded model
        for i in np.arange(num_worker):
            worker_id = i +1
            #load weigh from disk
            pretrained_dict = torch.load(self.federated_dir + str(worker_id) + 'cGANG.pth')
            model_dict = self.model_list[i].netG.state_dict() # load structrue from model

            # 1. filter out unnecessary keys
            pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict_trim)
            # 3. load the new state dict
            self.model_list[i].netG.load_state_dict(model_dict)

            # load weigh from disk
            pretrained_dict = torch.load(self.federated_dir + str(worker_id) + 'cGAND.pth')
            model_dict = self.model_list[i].netD.state_dict()  # load structrue from model

            # 1. filter out unnecessary keys
            pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict_trim)
            # 3. load the new state dict
            self.model_list[i].netD.load_state_dict(model_dict)


            # sum
            this_sdG=self.model_list[i].netG.state_dict()
            if (i==0):#ini
                average_sdG=this_sdG
            else:
                for key in average_sdG:
                    average_sdG[key] = (average_sdG[key] + this_sdG[key])
                    # sum
            this_sdD = self.model_list[i].netD.state_dict()
            if (i == 0):  # ini
                average_sdD = this_sdD
            else:
                for key in average_sdD:
                    average_sdD[key] = (average_sdD[key] + this_sdD[key])
        #AVERAGE
        for key in average_sdG:
            average_sdG[key] = (average_sdG[key])/ float(num_worker)
        self.average_model.netG.load_state_dict(average_sdG)
        for key in average_sdD:
            average_sdD[key] = (average_sdD[key])/ float(num_worker)
        self.average_model.netD.load_state_dict(average_sdD)
        torch.save(self.average_model.netD.state_dict(), self.federated_dir + "cGAND" + ".pth")
        torch.save(self.average_model.netG.state_dict(), self.federated_dir + "cGANG" + ".pth")
    def upload_local_files(self,upload_file_list):

        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.gdrive_id)}).GetList()

        for upload_file in upload_file_list:
            # check nessesary to delete the old file

            gfile = self.drive.CreateFile({'parents': [{'id': self.gdrive_id}]})
            gfile.SetContentFile(upload_file)
            gfile.Upload()
            print("one new uploaded")
            for file1 in file_list:
                if file1['title'] == upload_file:
                    file1.Delete()
                    print("one old deleted")

                else:
                    pass
        print("all uploaded")



    def load_model_local_worker(self,this_gdrive_id,worker_id):
        # get all the pth file from this worker

        while True:
            try:
                this_file_list = self.drive.ListFile(
                    {'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
                for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                    if file['fileExtension'] == 'pth':
                        _, original_name = os.path.split(file['title'])
                        print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j,
                                                                               len(this_file_list)))
                        file.GetContentFile(self.federated_dir + str(worker_id) + original_name)
                break
            except pydrive.files.ApiRequestError:
                print("Oops!  That was no model load..")

        pass
        return
    def load_all_local_models(self):
        num_worker = int(self.fed_json_data["machine number"])
        # load model whe every one is ready
        if (self.ready_worker_cnt >= num_worker):
            for i in np.arange(num_worker):
                worker_id = i + 1
                this_gdrive_id = self.fed_json_data["worker cloud id"][str(i + 1)]

                self.load_model_local_worker(this_gdrive_id, worker_id)
                pass
            print("all model loaded")
            # update this model when it is ready
        return
    # chech the status from worker, if it is ready, load the model
    def check_status_from_works(self):
        num_worker =int( self.fed_json_data["machine number"])
        self.ready_worker_cnt = 0
        self.loaded_worker_cnt =0
        for i in np.arange(num_worker):
            worker_id = i +1
            this_gdrive_id = self.fed_json_data["worker cloud id"][str(i+1)]

            while True:
                try:
                    this_file_list = self.drive.ListFile(
                        {'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
                    for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                        if file['fileExtension'] == 'json':
                            print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j,
                                                                                   len(this_file_list)))
                            this_json_worker_dir = self.federated_dir + "telecom/worker" + str(
                                worker_id) + ".json"
                            file.GetContentFile(this_json_worker_dir)
                    break
                except pydrive.files.ApiRequestError:
                    print("Oops!  That was no josn load..")


                    #load this work's json states
            with open(this_json_worker_dir) as f_dir:
                this_worker_json_data = JSON.load(f_dir)
            if (int(this_worker_json_data['minimal ready'])>0):
                self.ready_worker_cnt +=1
            if (this_worker_json_data['stage'] =='already_load_fed_model' ):
                self.loaded_worker_cnt+=1

        self.fed_json_data['ready machine number'] = str(self.ready_worker_cnt)
        with open(self.fed_json_dir, "w") as jsonFile:
            JSON.dump(self.fed_json_data, jsonFile)
        print("fed json status updated")
        return

# Path_dir, image_name = os.path.split(this_folder_list[0])

    def run(self):
        while(1):
            self.load_fed_json()
            self.check_status_from_works()
            if (self.ready_worker_cnt >= self.num_worker and self.fed_json_data['federated update'] == '0' and self.fed_json_data['stage']!='upload_waiting_remote_update' and self.fed_json_data['stage'] == 'fed_new_round' ):
                 self.load_all_local_models()
                 self.fed_average()
                 self.upload_local_files(self.upload_model_list)
                 self.fed_json_data['federated update'] = '1'
                 self.fed_json_data['stage'] = 'upload_waiting_remote_update'
                 self.write_fed_json()
                 print("fed json status updated")
                 self.upload_local_files(self.upload_json_list)
            if ( self.fed_json_data['stage'] == 'upload_waiting_remote_update'):
                if (self.loaded_worker_cnt>=self.num_worker):
                    # go back to inital stage
                    self.fed_json_data['federated update'] = '0'
                    self.fed_json_data['stage'] = 'fed_new_round'
                    self.write_fed_json()
                    print("fed json status updated")
                    self.upload_local_files(self.upload_json_list)
                    pass
            self.upload_local_files(self.upload_json_list)


if __name__ == '__main__':

    Handle = Cloud_Federated_Handler()
    # self test
    Handle.run()