%% Define paths, parameters etc

datasetPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\';
samplingPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\sampling\';
resultPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\result\';


datasetFiles = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm'};


uClasfs = {loglc2([],1e-3),loglc2([],1e-1),loglc2([],1),loglc2([],1e3), randomforestc2([],3), randomforestc2([],10), randomforestc2([],30), randomforestc2([],100)};

numBaggedCopies = 10;

numTrainTest = 5;
trainingSizesPerClass = [100, 300, 1000, 3000, 10000];
testSizePerClass = 10000;

flagOverwrite = 0; %Do not recompute results (only needed if there is a bug etc)


%% Generate samplings

for i = 1:length(datasetFiles)
    load(fullfile(datasetPath, [datasetFiles{i} '.mat']), 'y','id');
    
    numClasses = numel(unique(y));
    numSubjects = numel(unique(id));
    numTrainSubjects = floor(0.7*numSubjects);

    indexTrainBagId = nan(numTrainTest, numTrainSubjects);
    indexTestBagId = nan(numTrainTest, numSubjects-numTrainSubjects);
    
    for ntt=1:numTrainTest
        permutedBagId = id(randperm(numSubjects));

        indexTrainBagId = uint8(permutedBagId(1:numTrainSubjects));
        indexTestBagId = uint8(permutedBagId(numTrainSubjects+1:end));
        
        inTrainBag = ismember(id, indexTrainBagId);
        indexTrainInst = find(inTrainBag==1);
        indexTestInst = find(inTrainBag==0);
                
        indexSubTestInst = getBalancedSample(indexTestInst, y(indexTestInst), testSizePerClass);
        
        for ts = 1:numel(trainingSizesPerClass)
            fileName = fullfile(samplingPath, [datasetFiles{i} s '.mat']);
            
            
            if ~exist(fileName, 'file') || flagOverwrite == 1

                indexSubTrainInst = getBalancedSample(indexTrainInst, y(indexTrainInst), trainingSizesPerClass(ts));
                s = sprintf('_sampling_%d_test_%d_train_%d', ntt,testSizePerClass,trainingSizesPerClass(ts));
                save(fileName, 'indexTrainBagId','indexTestBagId', 'indexSubTrainInst','indexSubTestInst');
            else
                disp(['File ' fileName ' already exists, skipping']);
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
%%

for i=1:size(resClasf,1)
    resClasf(i,:) = tiedrank(resClasf(i,:));
end

whichData = repmat(trainingSizesPerClass(:),numTrainTest,1);
whichData = [whichData; whichData+1];

embedded = tsne(+resClasf);

%%
embeddedData = prdataset(embedded, whichData);

scatterd(embeddedData,'legend');




