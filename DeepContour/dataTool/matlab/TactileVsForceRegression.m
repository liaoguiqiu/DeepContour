
TrainingSet ={
%     'Endoscopic nomoving Phantom  + trans open -1 +stab 4th'; 
%     'Endoscopic nomoving Phantom  + trans open -3 +stab 4th';
%     'Endoscopic nomoving Phantom  + trans open -5 +stab 4th';
%     'Endoscopic nomoving Phantom + trans -90 0 alpha 995 +stab 4th';  
% 'Endoscopic Phantom + trans open -5 +stab'
% 'Endoscopic Phantom + trans -110 0 alpha 995 +stab 4th'
% 'Endoscopic Phantom + trans -90 0 alpha 995 +stab 4th'
% 'Endoscopic Phantom no trans open -1 +stab'
% 'Endoscopic Phantom no trans open -3 +stab'
% 'Endoscopic Phantom no trans open -5 +stab'
% 'Endoscopic Phantom No trqns -110 0 alpha 995 +stab'
% 'Endoscopic nomoving stiffer Phantom  + trans -90 0 alpha 995 +stab 4th'
% 'Endoscopic nomoving stiffer Phantom  + trans open -5 +stab 4th'
% 'Endoscopic Phantom stiffer no trans open -5 +stab'
% 'Endoscopic Phantom stiffer no trans -110 0 alpha 995 +stab'
% 'Endoscopic Phantom stiffer + trans open -5 +stab'
'Endoscopic Phantom stiffer+ trans -110 0 alpha 995 +stab'
     };
root = 'E:/DeeplearningData_jelly/dataset/label data/tactile_excel/';
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

[tactile_train,force_train,integrated] = DataLoader (root, TrainingSet);

Goodquality = integrated<0.05;
Poor_quality = integrated>0.1;
Middel_quality = (1-Goodquality).* (1-Poor_quality);
Quality_matrix =  Goodquality *1 +  Middel_quality *0.5  ;
Good_quality_rate = sum(Goodquality)/length (integrated)
Poor_quality_rate = sum(Poor_quality)/length (integrated);
Middle_quality_rate = sum(Middel_quality)/length (integrated);
Visible_rate = Good_quality_rate + Middle_quality_rate

force_larger10 =  force_train>10;
force_larger20 =  force_train>20;

force_larger10rate = sum(force_larger10)/length (force_larger20)
force_larger20rate = sum(force_larger20)/length (force_larger20)

figure(2)
imagesc (Quality_matrix);
function [tactile,force,integrated] = DataLoader(root,folder_list )
    cropflag = 0;
    crop_start =465;
    crop_end = 700;
    folder_num = length (folder_list);
    
    for k1 = 1: folder_num
        folder_list(k1)
        this_dir = strcat(root,string(folder_list(k1)) ,'/','error_buff.csv');
        rawdata =  csvread( this_dir);
        this_data = rawdata;
        if cropflag == 1
            this_data = rawdata(crop_start:crop_end,:);
        end
        
        if k1==1
            data_all  = this_data(3:end,:);
        else
            data_all  = [data_all; this_data(3:end,:)];
        end
    end
    
    tactile = data_all(:,3); % tactile only
    force = data_all(:,5); % force only
    integrated =  data_all(:,4); % integrated = distance - tactile 
    
    
end


