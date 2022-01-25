from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from working_dir_root import config_root,Output_root
import json as JSON

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
            json_data = JSON.load(f_dir)
        # gdrive_id = '1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o'
        self.gdrive_id  = json_data['worker cloud id']
        self.upload_file_list = [self.out_dir + 'cGAND.pth',
                            self.out_dir + 'cGANG.pth',
                            self.json_dir ]
   #  update the json files
    def json_initial(self):
        with open(self.json_dir) as f_dir:
            json_data = JSON.load(f_dir)
        newJson = json_data
        newJson['minimal ready']='0'
        newJson['last local update']='0'
        # shape  = data["shapes"]
        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(newJson, jsonFile)
        print("local json status initialized")
        return
    def json_update_after_epo(self):
        with open(self.json_dir) as f_dir:
            json_data = JSON.load(f_dir)
        newJson = json_data
        # this count will be clear after
        ready_cnt = int(json_data['minimal ready']) + 1
        newJson['minimal ready']= str(ready_cnt)
        update_cnt = int(json_data['last local update']) + 1
        newJson['last local update']=  str(update_cnt)

        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(newJson, jsonFile)
        print("local json status initialized")
        pass
    def json_update_after_download(self):
        with open(self.json_dir) as f_dir:
            json_data = JSON.load(f_dir)
        newJson = json_data
        # this count will be clear after update
        newJson['minimal ready'] = '0'
        with open(self.json_dir, "w") as jsonFile:
            JSON.dump(newJson, jsonFile)
        print("local json status updated")
        pass
    def upload_local_models(self):

        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.gdrive_id)}).GetList()

        for upload_file in self.upload_file_list:
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

    def test (self):
        # upload a list of files 
        upload_file_list = [ self.out_dir + 'cGAND_epoch_1.pth', 
                              self.out_dir + 'cGANG_epoch_1.pth',
                             self.out_dir + 'telecom/local_training_status.json']
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.gdrive_id)}).GetList()

        for upload_file in upload_file_list:
            # check nessesary to delete the old file 
            for file1 in file_list:
                if file1['title'] == upload_file:
                    file1.Delete()  
                else:                   
                    pass
            gfile = self.drive.CreateFile({'parents': [{'id': self.gdrive_id}]})
            gfile.SetContentFile(upload_file)
            gfile.Upload()
	        
        #  list all files from the specific folder in the google drive
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.gdrive_id)}).GetList()
        for file in file_list:
	        print('title: %s, id: %s' % (file['title'], file['id']))
        # Download the files from Google Drive
        for i, file in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
	        print('Downloading {} file from GDrive ({}/{})'.format(file['title'], i, len(file_list)))
	        file.GetContentFile(file['title'])

if __name__ == '__main__':

    API = Cloud_API()
    # self test
    API.test()