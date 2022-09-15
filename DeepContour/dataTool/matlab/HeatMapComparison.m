figure
plot (force_train); hold
plot (testtraining)

figure
all= [force_train, testtraining]
imagesc(all)
% in comparison to HeatMap comand, this one is more editable