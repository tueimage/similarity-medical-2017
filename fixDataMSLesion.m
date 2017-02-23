



datasetPath = ('C:\Users\Veronika\Dropbox\Papers\MICCAI 2017\data\');


load('C:\Users\Veronika\Dropbox\Papers\Brain\annegreet\FeaturesT1CHBUNCWMLbig10pWMLnoSF.mat');

id = 1:10;                %10 images  
id = repmat(id, 100000,1); %100000 samples per image
id = uint8(id(:));


%Only use CHB and UNC (RSS not public)
x=trnxCHB;
y=classx1CHB;

save(fullfile(datasetPath, 'MSLesionUpsampledCHB_copy.mat'), 'x', 'y', 'id'); 


x = trnxUNC;
y = classx1UNC;

save(fullfile(datasetPath, 'MSLesionUpsampledUNC_copy.mat'), 'x', 'y', 'id'); 