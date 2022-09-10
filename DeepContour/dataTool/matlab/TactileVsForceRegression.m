
TrainingSet =['Endoscopic nomoving Phantom  + trans open -1 +stab 4th',
            'Endoscopic nomoving Phantom  + trans open -3 +stab 4th'
            ]
root = 'E:/DeeplearningData_jelly/dataset/label data/tactile_excel/'
% Train_folder_num = length (TrainingSet(:,1));
%  
% for k1 = 1: Train_folder_num
%     this_dir = strcat(root,TrainingSet(k1,:) ,'/','error_buff.csv');
%     this_data =  csvread( this_dir);
%     if k1==1
%         data_all  = this_data
%     else
%         data_all  = [data_all; this_data]
%     end
% end

[tactile_train,force_train] = DataLoader (root, TrainingSet)
 
function [tactile,force] = DataLoader(root,folder_list )
    folder_num = length (folder_list(:,1));
 
    for k1 = 1: folder_num
        this_dir = strcat(root,folder_list(k1,:) ,'/','error_buff.csv');
        this_data =  csvread( this_dir);
        if k1==1
            data_all  = this_data(3:end,:)
        else
            data_all  = [data_all; this_data(3:end,:)]
        end
    end
    tactile = data_all(:,3)
    force = data_all(:,5)
end