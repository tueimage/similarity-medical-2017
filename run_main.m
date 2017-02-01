%% Define paths, parameters etc

datasetPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\';
samplingPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\sampling\';
resultPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\result\';
metaPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\bmpv\metadata\';

%datasetFiles = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB_copy','MSLesionUpsampledUNC_copy'};

datasetFiles = {'MSLesionUpsampledCHB_copy','MSLesionUpsampledUNC_copy'};



uClasfs = {nmc, ldc, qdc, loglc2([],1e-3),loglc2([],1e-1),loglc2([],1),loglc2([],1e3)}; % randomforestc2([],3)*classc, randomforestc2([],10)*classc, randomforestc2([],30)*classc, randomforestc2([],100)*classc};


%uClasfs = {randomforestc2([],3)*classc, randomforestc2([],10)*classc, randomforestc2([],30)*classc, randomforestc2([],100)*classc};


numTrainTest = 10;
trainingSizesPerClass = [100, 300, 1000, 3000, 10000];
testSizePerClass = 10000;

flagOverwrite = 1; %If = 0, do not recompute results (only needed if there is a bug etc)


%% Generate samplings

for i = 1:length(datasetFiles)
    load(fullfile(datasetPath, [datasetFiles{i} '.mat']), 'y','id');
    
    numClasses = numel(unique(y));
    uniqueSubjects = unique(id);
    numSubjects = numel(uniqueSubjects);
    numTrainSubjects = floor(0.7*numSubjects);

    indexTrainBagId = nan(numTrainTest, numTrainSubjects);
    indexTestBagId = nan(numTrainTest, numSubjects-numTrainSubjects);
    
    for ntt=1:numTrainTest
        permutedBagId = uniqueSubjects(randperm(numSubjects));

        indexTrainBagId = uint8(permutedBagId(1:numTrainSubjects));
        indexTestBagId = uint8(permutedBagId(numTrainSubjects+1:end));
        
        inTrainBag = ismember(id, indexTrainBagId);
        indexTrainInst = find(inTrainBag==1);
        indexTestInst = find(inTrainBag==0);
                
        indexSubTestInst = getBalancedSample(indexTestInst, y(indexTestInst), testSizePerClass);
        
        for ts = 1:numel(trainingSizesPerClass)
            
            s = sprintf('_sampling_%d_test_%d_train_%d', ntt,testSizePerClass,trainingSizesPerClass(ts));
            fileName = fullfile(samplingPath, [datasetFiles{i} s '.mat']);
            
            
            if ~exist(fileName, 'file') || flagOverwrite == 1

                indexSubTrainInst = getBalancedSample(indexTrainInst, y(indexTrainInst), trainingSizesPerClass(ts));
                
                try
                    assert(numel(unique(y(indexSubTrainInst))) == numClasses);
                catch
                        keyboard
                end
                save(fileName, 'indexTrainBagId','indexTestBagId', 'indexSubTrainInst','indexSubTestInst');
            else
                %disp(['File ' fileName ' already exists, skipping']);
            end

        end    
   
  end
end
%% Run the classifiers

resClasf = [];

for i = 1:length(datasetFiles)
    samplingFiles = dir(fullfile(samplingPath, [datasetFiles{i} '_sampling*']));
    
    load(fullfile(datasetPath, [datasetFiles{i} '.mat']), 'x','y','id');
    x = double(x); %Otherwise PRTools gets confused
    
   for j=1:numel(samplingFiles)
       load(fullfile(samplingPath, samplingFiles(j).name));
       
       trainData = prdataset(x(indexSubTrainInst,:), y(indexSubTrainInst));
       testData = prdataset(x(indexSubTestInst,:), y(indexSubTestInst));
       
       for u=1:length(uClasfs)
           
           uClasf = scalem([],'variance')*uClasfs{u};
           fileName = fullfile(resultPath, [samplingFiles(j).name '_' getname(uClasf) '.mat']);
           
           if ~exist(fileName, 'file') || flagOverwrite == 1
                trClasf = trainData*uClasf;
                resData = testData*trClasf;

                save(fileName, 'resData'); %Save posterior probabilities so we can compute any metric later
           else
                disp(['File ' fileName ' already exists, skipping']);
           end
       end
    end
 end
%% Create meta-dataset


datasetFiles = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB_copy','MSLesionUpsampledUNC_copy'};


for i = 1:length(datasetFiles)
    samplingFiles = dir(fullfile(samplingPath, [datasetFiles{i} '_sampling*']));
    
    metaLabelName = [];
    metaLabelTrainSize = [];
    metaDataOfDatabase = [];
    
   for j=1:numel(samplingFiles)
       
       fileNameParts = strsplit(samplingFiles(j).name, '_');
       metaLabelName = strvcat(metaLabelName, fileNameParts{1});              %TODO: Notice the hard-coded assumption about the file name structure!
       fileNameParts = strsplit(fileNameParts{7},'.');
       metaLabelTrainSize = [metaLabelTrainSize; str2num(fileNameParts{1})];
       
       acc=nan(1, length(uClasfs)); %initialize vectors for metadata of this particular dataset
      
       
       resultFiles = dir(fullfile(resultPath, [samplingFiles(j).name '*']));
       %assert(length(resultFiles) == length(uClasfs));
       
       for u=1:length(resultFiles)
       
       
           %fileName = fullfile(resultPath, [samplingFiles(j).name '_' getname(uClasfs{u}) '.mat']);
           fileName = fullfile(resultPath, [resultFiles(u).name]);
           
           load(fileName, 'resData'); 
           acc(u) = 1-testc(resData);             %PRTools reports errors
       end
       metaDataOfDatabase = [metaDataOfDatabase; acc];
   end
   save(fullfile(metaPath, datasetFiles{i}), 'metaDataOfDatabase', 'uClasfs','metaLabelName', 'metaLabelTrainSize');
end

%% Get meta-dataset, embed it 

%datasetFiles = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB_copy','MSLesionUpsampledUNC_copy'};

datasetFilesToInclude = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB_copy','MSLesionUpsampledUNC_copy'};




metaDataAll = [];
metaLabelNameAll = [];

for i = 1:length(datasetFilesToInclude)
   load(fullfile(metaPath, datasetFilesToInclude{i}), 'metaDataOfDatabase', 'uClasfs','metaLabelName');
   metaDataAll = [metaDataAll; metaDataOfDatabase];
   metaLabelNameAll = strvcat(metaLabelNameAll, metaLabelName);
end
metaDataRank = nan(size(metaDataAll));



%Create extra dataset based on rank
for i=1:size(metaDataAll,1)
    %metaDataRank(i,:) = tiedrank(metaDataAll(i,:));
    
    perf = tiedrank(metaDataAll(i,:));
    metaDataRank(i,:) = (perf-mean(perf))./std(perf);
end

%%
close all;
%tsneAcc = tsne(+metaDataAll); 
%tsneAcc = prdataset(tsneAcc, metaLabelNameAll);


tsneRank = tsne(+metaDataRank);  %TODO: tSNE is stochastic, need to actually do this a few times and select embedding with lowest loss
tsneRank = prdataset(tsneRank, metaLabelNameAll);



ml = unique(metaLabelNameAll, 'rows');
colors = [0.75 0.5 0; 0.75 0 0.5; 0 0.75 0.5; 0.5 0.75 0; 0.5 0 0.75; 0 0.5 0.75; 0.5 0.5 0.5];
    

for i=1:size(ml,1)
    c = seldat(tsneAcc, ml(i,:)); 
    scatter(+c(:,1), +c(:,2), 'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', colors(i,:));
    hold on;
end
legend(ml);

%%
%Compute distances, set diagonal to 0 so MDS doesn't complain
distMetaDataAll = sqeucldistm(+metaDataRank,+metaDataRank);
distMetaDataAll(eye(size(distMetaDataAll))==1) = 0;

mdsAcc = distMetaDataAll*mds(distMetaDataAll);
mdsAcc = prdataset(+mdsAcc, metaLabelNameAll);


for i=1:size(ml,1)
    c = seldat(mdsAcc, ml(i,:)); 
    scatter(+c(:,1), +c(:,2), 'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', colors(i,:));
    hold on;
end
legend(ml);


