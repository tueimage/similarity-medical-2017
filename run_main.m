%% Define paths, parameters etc
clear all;

datasetPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\';
samplingPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\sampling\';
resultPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\result\';
metaPath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\bmpv\metadata\';
figurePath = 'C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\bmpv\figure\';

datasetFiles = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB','MSLesionUpsampledUNC', 'Pim'};
%datasetFiles = {'Pim'};




uClasfs = {nmc, ldc, qdc, loglc2([],1), randomforestc2([],1)*classc, fnnc}; 

numTrainTest = 20;
trainingSizes = [100, 300, 1000, 3000, 10000];
testSize = 10000;

flagOverwrite = 0; %If = 0, do not recompute results (only needed if there is a bug etc)


%For plots etc
[niceNameMap, niceColorMap] = getNiceDataNames(datasetFiles);


%% Generate samplings

for i = 1:length(datasetFiles)
    load(fullfile(datasetPath, [datasetFiles{i} '.mat']), 'y','id');
    
    numClasses = numel(unique(y));
    uniqueSubjects = unique(id);
    numSubjects = numel(uniqueSubjects);
    numTrainSubjects = floor(0.7*numSubjects);
    
    for ntt=1:numTrainTest
        
        labelsOK = 0;
        
        while(labelsOK == 0) 
            permutedBagId = uniqueSubjects(randperm(numSubjects));
            indexTrainBagId = permutedBagId(1:numTrainSubjects);
            indexTestBagId = permutedBagId(numTrainSubjects+1:end);

            inTrainBag = ismember(id, indexTrainBagId);
            indexTrainInst = find(inTrainBag==1);
            indexTestInst = find(inTrainBag==0);

            indexSubTestInst = getBalancedSample(indexTestInst, y(indexTestInst), testSize);
            indexSubTrainInst = getBalancedSample(indexTrainInst, y(indexTrainInst), trainingSizes(end));
            
            if numel(unique( y(indexSubTestInst,:)   ))==numClasses && numel(unique( y(indexSubTrainInst,:)   ))==numClasses
                labelsOK = 1;
            end
        end
        
        
        for ts = 1:numel(trainingSizes)
            
            s = sprintf('_sampling_%d_test_%d_train_%d', ntt,testSize,trainingSizes(ts));
            fileName = fullfile(samplingPath, [datasetFiles{i} s '.mat']);
            
            
            if ~exist(fileName, 'file') || flagOverwrite == 1

                indexSubTrainInst = getBalancedSample(indexTrainInst, y(indexTrainInst), trainingSizes(ts));
                
                
                assert(numel(unique( y(indexSubTrainInst,:)   ))==numClasses)
        
                
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
       
       assert(numel(unique(y(indexSubTrainInst)))==numel(unique(y(indexSubTestInst))))
       
       for u=1:length(uClasfs)
           
           uClasf = scalem([],'variance')*uClasfs{u};
           fileName = fullfile(resultPath, [samplingFiles(j).name '_' getname(uClasf) '.mat']);
           
           if ~exist(fileName, 'file') || flagOverwrite == 1
                trClasf = trainData*uClasf;
                resData = testData*trClasf;

                save(fileName, 'resData'); %Save posterior probabilities so we can compute any metric later
           else
                %disp(['File ' fileName ' already exists, skipping']);
           end
       end
    end
 end
%% Create meta-dataset

for i = 1:length(datasetFiles)
    
    for j=1:numTrainTest
        
       datasetMetaData=nan(numel(trainingSizes), length(uClasfs)); %initialize vectors for metadata of this particular dataset
    
       for k=1:numel(trainingSizes)

           resultFiles = dir(fullfile(resultPath, [datasetFiles{i} '_sampling_' num2str(j) '_test_' num2str(testSize) '_train_' num2str(trainingSizes(k)) '.*']));
           
           for u=1:length(resultFiles)
               fileName = fullfile(resultPath, [resultFiles(u).name]);

               load(fileName, 'resData'); 
               datasetMetaData(k,u) = 1-testc(resData);             %PRTools reports errors
           end
       end
       
       save(fullfile(metaPath, [datasetFiles{i} '_sampling_' num2str(j) '.mat']), 'datasetMetaData');
   end
end

%% Get meta-dataset, embed it 

allMetaAcc = [];
allMetaRank = [];
allMetaScaled = [];

allMetaLabel = [];


for i = 1:length(datasetFiles)
   for j=1:numTrainTest 
       load(fullfile(metaPath, [datasetFiles{i} '_sampling_' num2str(j) '.mat']), 'datasetMetaData'); 
       
       %datasetMetaData is numel(trainingSizesPerClass) x length(uClasfs), need to reduce to 1xlength(uClasfs)
       
       datasetMetaRank = nan(size(datasetMetaData));
       datasetMetaScaled = nan(size(datasetMetaData));
       
       %Convert performances to ranks (for R), scale performances (for S)
       for k=1:size(datasetMetaData,1)
            datasetMetaRank(k,:) = tiedrank(datasetMetaData(k,:));
            
            if std(datasetMetaData(k,:)) == 0
                datasetMetaScaled(k,:) = zeros(size(datasetMetaData(k,:)));
            else
                datasetMetaScaled(k,:) = (datasetMetaData(k,:) - mean(datasetMetaData(k,:))) / std(datasetMetaData(k,:));
            end
       end
        %Average over different training sizes
       datasetMetaAcc = mean(datasetMetaData,1);
       datasetMetaRank = mean(datasetMetaRank,1);
       datasetMetaScaled = mean(datasetMetaScaled,1);
       
       %Append to overall meta dataset
       allMetaAcc = [allMetaAcc; datasetMetaAcc];
       allMetaRank = [allMetaRank; datasetMetaRank];
       allMetaScaled = [allMetaScaled; datasetMetaScaled];
       
       allMetaLabel = strvcat(allMetaLabel, datasetFiles{i});
   end
end

%% tSNE and MDS embeddings

perplexity = 5; %More samples = higher perplexity
numTries = 10;  %tSNE is stochastic, so will do embedding a few times and select embedding with lowest loss

embeddingSeed = 1;

rand('seed', embeddingSeed);
tsneAcc = getEmbeddingTSNE(allMetaAcc, perplexity, numTries);

rand('seed', embeddingSeed);
tsneRank = getEmbeddingTSNE(allMetaRank, perplexity, numTries);

rand('seed', embeddingSeed);
tsneScaled = getEmbeddingTSNE(allMetaScaled, perplexity, numTries);

mdsAcc = getEmbeddingMDS(allMetaAcc);
mdsRank = getEmbeddingMDS(allMetaRank);
mdsScaled = getEmbeddingMDS(allMetaScaled);


save(fullfile(figurePath, ['embedding.mat']), 'allMetaAcc', 'allMetaRank', 'allMetaScaled', 'tsneAcc', 'tsneRank', 'tsneScaled', 'mdsAcc', 'mdsRank', 'mdsScaled', 'embeddingSeed','perplexity', 'numTries');

%% Plots tSNE and MDS
[niceNameMap, niceColorMap] = getNiceDataNames(datasetFiles);


load(fullfile(figurePath, ['embedding.mat']), 'allMetaAcc', 'allMetaRank', 'allMetaScaled', 'tsneAcc', 'tsneRank', 'tsneScaled', 'mdsAcc', 'mdsRank', 'mdsScaled', 'embeddingSeed','perplexity', 'numTries');


close all;
figure(1);
ha=tight_subplot(3,2, [0.05 0.02], 0.05,0.02);

axes(ha(1));
legendNames = plotEmbedding(tsneAcc, allMetaLabel, niceNameMap, niceColorMap);
title('t-SNE Accuracy', 'interpreter', 'latex');

axes(ha(3));
plotEmbedding(tsneRank, allMetaLabel, niceNameMap, niceColorMap);
title('t-SNE Rank','interpreter', 'latex');

axes(ha(5));
plotEmbedding(tsneScaled, allMetaLabel, niceNameMap, niceColorMap);
title('t-SNE Norm', 'interpreter', 'latex');


axes(ha(2));
plotEmbedding(mdsAcc, allMetaLabel,niceNameMap,niceColorMap,'legend');
title('MDS Accuracy', 'interpreter', 'latex');


axes(ha(4));
plotEmbedding(mdsRank, allMetaLabel, niceNameMap,niceColorMap);
title('MDS Rank', 'interpreter', 'latex');

axes(ha(6));
plotEmbedding(mdsScaled, allMetaLabel, niceNameMap,niceColorMap);
title('MDS Norm', 'interpreter', 'latex');


set(gcf,'Color',[1 1 1])
set(gcf, 'Position', [100 100 800 1200]);

saveas(gcf,fullfile(figurePath, ['embedding.fig']));
export_fig(fullfile(figurePath, ['embedding.pdf']));

%% Can we predict a dataset's label based on its meta data? 
load(fullfile(figurePath, ['embedding.mat']), 'tsneAcc', 'tsneRank', 'tsneScaled', 'mdsAcc', 'mdsRank', 'mdsScaled', 'allMetaLabel', 'niceNameMap');


metaTrainSizes = [5 10 20 40 80];
metaTries = 5;
metaClasf = {scalem([],'variance')*knnc([],1), scalem([],'variance')*loglc2};


rawAccX = prdataset(allMetaAcc, allMetaLabel);
rawRankX = prdataset(allMetaRank, allMetaLabel);
rawScaledX = prdataset(allMetaScaled, allMetaLabel);

tsneAccX = prdataset(tsneAcc, allMetaLabel);
tsneRankX = prdataset(tsneRank, allMetaLabel);
tsneScaledX = prdataset(tsneScaled, allMetaLabel);

mdsAccX = prdataset(mdsAcc, allMetaLabel);
mdsRankX = prdataset(mdsRank, allMetaLabel);
mdsScaledX = prdataset(mdsScaled, allMetaLabel);


rawAccErr = cleval_with_confmat(rawAccX, metaClasf, metaTrainSizes, metaTries);
rawRankErr = cleval_with_confmat(rawRankX, metaClasf, metaTrainSizes, metaTries);
rawScaledErr = cleval_with_confmat(rawScaledX, metaClasf, metaTrainSizes, metaTries);

tsneAccErr = cleval_with_confmat(tsneAccX, metaClasf, metaTrainSizes, metaTries);
tsneRankErr = cleval_with_confmat(tsneRankX, metaClasf, metaTrainSizes, metaTries);
tsneScaledErr = cleval_with_confmat(tsneScaledX, metaClasf, metaTrainSizes, metaTries);

mdsAccErr = cleval_with_confmat(mdsAccX, metaClasf, metaTrainSizes, metaTries);
mdsRankErr = cleval_with_confmat(mdsRankX, metaClasf, metaTrainSizes, metaTries);
mdsScaledErr = cleval_with_confmat(mdsScaledX, metaClasf, metaTrainSizes, metaTries);



save(fullfile(figurePath, ['learncurve.mat']), 'metaTrainSizes', 'rawAccErr', 'rawRankErr', 'rawScaledErr', 'tsneAccErr', 'tsneRankErr', 'tsneScaledErr', 'mdsAccErr', 'mdsRankErr', 'mdsScaledErr','whichClasf');

%% Plot learning curves

load(fullfile(figurePath, ['learncurve.mat']), 'metaTrainSizes', 'rawAccErr', 'rawRankErr', 'rawScaledErr', 'tsneAccErr', 'tsneRankErr', 'tsneScaledErr', 'mdsAccErr', 'mdsRankErr', 'mdsScaledErr','whichClasf');

whichClasf = 1; %1 is 1-NN, 2 is Logistic

c1 = [0.75 0 0.25];
c2 = [0 0.5 0.75];
c3 = [0.5 0.75 0];
lw = 1.5;


figure(3);
%plot(metaTrainSizes, rawAccErr.error(whichClasf,:), 'o-', 'Color', c1, 'LineWidth', lw); hold on;
%plot(metaTrainSizes, rawRankErr.error(whichClasf,:), 'o-', 'Color', c2, 'LineWidth', lw ); hold on;
%plot(metaTrainSizes, rawScaledErr.error(whichClasf,:), 'o-', 'Color', c3,'LineWidth', lw); hold on;
plot(metaTrainSizes, tsneAccErr.error(whichClasf,:), 'v-', 'Color', c1, 'LineWidth', lw); hold on;
plot(metaTrainSizes, tsneRankErr.error(whichClasf,:), 'v-', 'Color', c2, 'LineWidth', lw); hold on;
plot(metaTrainSizes, tsneScaledErr.error(whichClasf,:), 'v-', 'Color', c3, 'LineWidth', lw); hold on;
plot(metaTrainSizes, mdsAccErr.error(whichClasf,:), 's--', 'Color', c1, 'LineWidth', lw); hold on;
plot(metaTrainSizes, mdsRankErr.error(whichClasf,:), 's--', 'Color', c2, 'LineWidth', lw); hold on;
plot(metaTrainSizes, mdsScaledErr.error(whichClasf,:), 's--', 'Color', c3, 'LineWidth', lw); hold on;

ylim([0.1 0.8]);
title('Learning curves', 'interpreter', 'latex');
legend({'Acc tSNE', 'Rank tSNE', 'Norm tSNE', 'Acc MDS', 'Rank MDS', 'Norm MDS'}, 'interpreter', 'latex');
%legend({'Acc all', 'Rank all', 'Scale all', 'Acc tSNE', 'Rank tSNE', 'Scale tSNE', 'Acc MDS', 'Rank MDS', 'Scale MDS'}, 'interpreter', 'latex');
ylabel('Error');
xlabel('Training size');

set(gca,'FontName','Georgia','FontSize',12);
set(gca,'Color',[1 1 1])
set(gcf,'Color',[1 1 1])

save(fullfile(figurePath, ['learncurve.mat']), 'metaTrainSizes', 'rawAccErr', 'rawRankErr', 'rawScaledErr', 'tsneAccErr', 'tsneRankErr', 'tsneScaledErr', 'mdsAccErr', 'mdsRankErr', 'mdsScaledErr','whichClasf');
saveas(gcf,fullfile(figurePath, ['learncurve.fig']));
export_fig(fullfile(figurePath, ['learncurve.pdf']));


%%
close all;

load(fullfile(figurePath, ['learncurve.mat']), 'metaTrainSizes', 'tsneScaledErr', 'whichClasf');

confusionMatrix = tsneScaledErr.confmat(:,:,whichClasf,numel(metaTrainSizes)); %largest training size
N = size(confusionMatrix,1);
totals = repmat(sum(confusionMatrix,2), 1, size(confusionMatrix,2)); %Rows are the true class. Therefore we need to sum over the columns, so that the numbers per row add up to 1. 
confusionMatrix = confusionMatrix ./ totals*100;

c1 = cbrewer('seq', 'Reds', 8);              
c2 = cbrewer('seq', 'YlGn', 8);
c = [c1; c2];
colormap(c);

imagesc(confusionMatrix);

textStrings=arrayfun(@(x) num2str(x,'%2.1f'), confusionMatrix(:), 'UniformOutput', false)

[x,y] = meshgrid(1:N, 1:N);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center', 'interpreter', 'latex', 'FontSize', 14);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(0,1,3); 
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
set(gca, 'yTickLabel', legendNames);
xticklabel_rotate([1:N],45,legendNames,'interpreter','latex')

title('$\downarrow$ True, $\rightarrow$ Predicted', 'interpreter','latex');
set(gca,'FontName', 'Georgia');
set(gcf,'Color',[1 1 1])

save(fullfile(figurePath, ['confmat.mat']), 'confusionMatrix');
saveas(gcf,fullfile(figurePath, ['confmat.fig']));
export_fig(fullfile(figurePath, ['confmat.pdf']));



