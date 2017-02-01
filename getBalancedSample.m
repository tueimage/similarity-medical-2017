function sampledix = getBalancedSample(initialix, y, sizePerClass)



classLabels = unique(y);
numClasses = numel(classLabels);

sampledix=[];

for i=1:numClasses
   
    samplesOfClass = initialix(y==classLabels(i));
    
    whichSamples = randi(numel(samplesOfClass),sizePerClass,1);
    
    sampledFromClass = samplesOfClass(whichSamples);
    sampledFromClass = sampledFromClass(:);
    sampledix = [sampledix; sampledFromClass];
end




