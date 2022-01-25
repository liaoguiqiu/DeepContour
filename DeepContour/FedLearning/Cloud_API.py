from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from working_dir_root import config_root,Output_root
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
   # add new step of all signals
    def func(self,x):
        pass
    def test (self):
        # upload a list of files 
        upload_file_list = [ self.out_dir + 'cGAND_epoch_1.pth', 
                              self.out_dir + 'cGANG_epoch_1.pth',
                             self.out_dir + 'telecom/local_training_status.json']
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format('1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o')}).GetList()

        for upload_file in upload_file_list:
            # check nessesary to delete the old file 
            for file1 in file_list:
                if file1['title'] == upload_file:
                    file1.Delete()  
                else:                   
                    pass
            gfile = self.drive.CreateFile({'parents': [{'id': '1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o'}]})
            gfile.SetContentFile(upload_file)
            gfile.Upload()
	        
        #  list all files from the specific folder in the google drive
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format('1kD2t08Df5YW9_oNyZhQgGmoTPYEH2x9o')}).GetList()
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