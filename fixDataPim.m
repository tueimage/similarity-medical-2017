


% datasetPaths = {'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\Pim_Train\', 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\Pim_Test\'};
% 
% for j=1:length(datasetPaths)
%     allFiles = dir(fullfile(datasetPaths{j},'features*'));
%     for i=1:length(allFiles)
% 
%         vars=load(fullfile(datasetPaths{j},allFiles(i).name));
%         varName = allFiles(i).name(1:end-4); %Every variable is named the same as the file it is stored in
% 
%         x = getfield(vars,varName);
% 
%         y = x(:,1);             %the first element of each row is the label (0-6), then the coordinates (x,y,z) if we would want to go back to image space, the rest are the features
%         locations = x(:,2:4); 
%         x = x(:,5:end);
%         save(fullfile(datasetPaths{j},allFiles(i).name), 'x', 'y', 'locations'); 
%     end
% end


%%
%TODO: run above for Pim_Test

%Create script for including IDs, combining different files into a single
%file (will need to adjust code if train/test subjects must stay that way)

%datasetPaths = {'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\Pim_Train\', 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\Pim_Test\'};


datasetPaths = {'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\Pim_Test\'};



allX = [];
allY = [];
id=[];

for j=1:length(datasetPaths)
     allFiles = dir(fullfile(datasetPaths{j},'features*'));
     for i=1:length(allFiles)

         bagid = str2num(allFiles(i).name(9:12)); %featuresXXXX
         
         load(fullfile(datasetPaths{j},allFiles(i).name), 'x','y'); %File now has: x, y, locations. Not using locations for now
        
        
         %Already subsample
         indexInst = 1:size(x,1);
         indexSubInst = getBalancedSample(indexInst, y(indexInst), 1000);
         allX = [allX; x(indexSubInst,:)];
         allY = [allY; y(indexSubInst)];
                  
         id = [id; repmat(bagid, numel(indexSubInst), 1)];
         
     end
end
x=allX;
y=allY;
save(fullfile('C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data','Pim'), 'x', 'y', 'id'); 