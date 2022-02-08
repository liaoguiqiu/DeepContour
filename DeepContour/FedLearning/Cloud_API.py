from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from working_dir_root import config_root,Output_root
import json as JSON
import  os
import pydrive
import  oauth2client
class Cloud_API(object):
    def __init__(self ):
        self.credential_dir = config_root + "client_secrets.json" 
        gauth = GoogleAuth()    
        gauth.LocalWebserverAuth( )
        # Create local webserver and auto handles authentication.
        #auth_url = self.gauth.GetAuthUrl() # Create authentication url user needs to visit
        #code = AskUserToVisitLinkAndGiveCode(auth_url) # Your customized authentication flow
        #self.gauth.Auth(code) # Authorize and build service from the code
        self.drive = GoogleDrive(gauth)  
        self.out_dir  = Output_root + "CEnet_trained/"
        self.json_dir = self.out_dir + 'telecom/local_training_status.json' # load json file for drive/cloud and local information
        with open(self.json_dir) as f_dir:
            self.json_data = JSON.load(f_dir)
        # gdrive_id = '1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o'
        self.gdrive_id  = self.json_data['worker cloud id']


        self.upload_file_list = [self.out_dir + 'cGAND.pth',
                            self.out_dir + 'cGANG.pth',
                            self.json_dir ]
        self.upload_model_list =  [self.out_dir + 'cGAND.pth',
                            self.out_dir + 'cGANG.pth']
        self.upload_json_list =  [ self.json_dir]
   #  update the json files
    def load_json(self):
        with open(self.json_dir) as f_dir:
            self.json_data = JSON.load(f_dir)
            # self.ready_worker_cnt =  self.fed_json_data['ready machine number']
    def write_json(self):
        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(self.json_data, jsonFile)
    def load_fed_model(self):
        # self.load_json()
        self.check_fed_json()
        this_gdrive_id = self.json_data["federated cloud id"]

        if self.fed_json_data['federated update']=='1':

            while True:
                try:
                    this_file_list = self.drive.ListFile(
                        {'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
                    for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                        if file['fileExtension'] == 'pth':
                            _, original_name = os.path.split(file['title'])
                            print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j,
                                                                                   len(this_file_list)))
                            file.GetContentFile(self.out_dir +   original_name)
                    break
                except (pydrive.files.ApiRequestError,oauth2client.client.HttpAccessTokenRefreshError):
                    print("Oops!  That was no json load..")


            pass
            self.json_data['stage'] = 'downloaded_new_model'
            self.write_json()
        return
    def check_fed_json(self):
        this_gdrive_id = self.json_data["federated cloud id"]


        while True:
            try:
                this_file_list = self.drive.ListFile(
                    {'q': "'{}' in parents and trashed=false".format(this_gdrive_id)}).GetList()
                for j, file in enumerate(sorted(this_file_list, key=lambda x: x['title']), start=1):
                    if file['fileExtension'] == 'json':
                        print('Downloading {} file from GDrive ({}/{})'.format(file['title'], j,
                                                                               len(this_file_list)))
                        this_json_dir = self.out_dir + "telecom/federated_learning_status.json"
                        file.GetContentFile(this_json_dir)
                break
            except (pydrive.files.ApiRequestError,oauth2client.client.HttpAccessTokenRefreshError):
                print("Oops!  That was no json load..")

        # load this work's json states
        with open(this_json_dir) as f_dir:
            self.fed_json_data = JSON.load(f_dir)




        print("fed json status updated")
    def json_initial(self):
        with open(self.json_dir) as f_dir:
            json_data = JSON.load(f_dir)
        newJson = json_data
        newJson['minimal ready']='0'
        newJson['last local update']='0'
        newJson['federated loaded'] = '0'

        # shape  = data["shapes"]
        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(newJson, jsonFile)
        print("local json status initialized")
        return
    def json_update_after_epo(self):
        with open(self.json_dir) as f_dir:
            self. json_data = JSON.load(f_dir)

        # this count will be clear after
        ready_cnt = int(self. json_data['minimal ready']) + 1
        self. json_data['minimal ready']= str(ready_cnt)
        update_cnt = int(self. json_data['last local update']) + 1
        self. json_data['last local update']=  str(update_cnt)

        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(self. json_data, jsonFile)
        print("local json status initialized")
        pass
    def json_update_after_newround(self):
        # with open(self.json_dir) as f_dir:
        #     self. json_data = JSON.load(f_dir)

        # this count will be clear after update
        self. json_data['minimal ready'] = '0'
        self. json_data['federated loaded'] = '0'
        self.json_data['stage'] = "local_new_round"

        # with open(self.json_dir, "w") as jsonFile:
        #     JSON.dump(self. json_data, jsonFile)
        print("local json status updated new round")
        pass
    def upload_local_files(self,upload_file_list):
        while True:
            try:
                file_list = self.drive.ListFile(
                    {'q': "'{}' in parents and trashed=false".format(self.gdrive_id)}).GetList()

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
                break
            except (pydrive.files.ApiRequestError,oauth2client.client.HttpAccessTokenRefreshError):
                print("Oops!  That was no model load..")
        return


    def test (self):
        # upload a list of files 
       return

if __name__ == '__main__':

    API = Cloud_API()
    # self test
    API.test()