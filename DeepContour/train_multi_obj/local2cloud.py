from model import CE_build3 # the mmodel
from FedLearning.Cloud_API import Cloud_API
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
class Local2Cloud(object):
    def __init__(self, pth_save_dir):
        # self.model = model
        # self.CE_Nets = model
        self. cloud_local_infer = Cloud_API()
        self.pth_save_dir = pth_save_dir
    def reset_model_para(self,model, name='cGANG'):
        pretrained_dict = torch.load(self.pth_save_dir + name + '.pth')
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_trim)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        return model
    def check_load_global_cloud(self,CE_Nets):

        self.cloud_local_infer.load_json()
        self.cloud_local_infer.check_fed_json()
        if (self.cloud_local_infer.fed_json_data['stage'] == "fed_new_round" and self.cloud_local_infer.json_data[
            'stage'] != "local_new_round" and self.cloud_local_infer.json_data['stage'] != "waiting_fed_update"):
            self.cloud_local_infer.json_update_after_newround()
            self.cloud_local_infer.write_json()
            self.cloud_local_infer.upload_local_files(self.cloud_local_infer.upload_json_list)
            # check to update

        self.cloud_local_infer.load_json()

        if (self.cloud_local_infer.fed_json_data['stage'] == 'upload_waiting_remote_update'):
            self.cloud_local_infer.load_fed_model()

            # stage =

            CE_Nets.netG = self.reset_model_para(CE_Nets.netG, name='cGANG')
            CE_Nets.netD = self.reset_model_para(CE_Nets.netD, name='cGAND')
            self.cloud_local_infer.json_data['stage'] = 'already_load_fed_model'
            self.cloud_local_infer.write_json()
        self.cloud_local_infer.upload_local_files(self.cloud_local_infer.upload_json_list)

        return CE_Nets

    def save_local_cloud(self, CE_Nets):
        self.cloud_local_infer.load_json()
        self.cloud_local_infer.check_fed_json()
        if (self.cloud_local_infer.fed_json_data['stage'] == "fed_new_round" and self.cloud_local_infer.json_data[
            'stage'] != "local_new_round" and self.cloud_local_infer.json_data['stage'] != "waiting_fed_update"):
            self.cloud_local_infer.json_data['stage'] = "local_new_round"
            self.cloud_local_infer.json_update_after_newround()
            self.cloud_local_infer.write_json()
            self.cloud_local_infer.upload_local_files(self.cloud_local_infer.upload_json_list)

        if (self.cloud_local_infer.json_data['stage'] == "local_new_round" or self.cloud_local_infer.json_data[
            'stage'] == "waiting_fed_update"):
            # save the latest model as the same name
            torch.save(CE_Nets.netG.state_dict(), self.pth_save_dir + "cGANG" + ".pth")
            torch.save(CE_Nets.netE.state_dict(), self.pth_save_dir + "cGANE" + ".pth")
            torch.save(CE_Nets.netD.state_dict(), self.pth_save_dir + "cGAND" + ".pth")
            # API interaction
            self.cloud_local_infer.json_update_after_epo()
            self.cloud_local_infer.upload_local_files(self.cloud_local_infer.upload_model_list)
            self.cloud_local_infer.json_data['stage'] = "waiting_fed_update"
            self.cloud_local_infer.write_json()
            self.cloud_local_infer.upload_local_files(self.cloud_local_infer.upload_json_list)