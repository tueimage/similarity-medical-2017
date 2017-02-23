function [niceNameMap, niceColorMap] = getNiceDataNames(dataNames)


numDatasets = length(dataNames);


keys = {'AVClassificationDRIVE','VesselSegmentationDRIVE','MitkoNorm','MitkoNoNorm','MicroaneurysmDetectionEophtha','MSLesionUpsampledCHB','MSLesionUpsampledUNC', 'Pim'};
values = {'ArteryVein','Vessel', 'Mitosis', 'MitosisNorm', 'Microaneurysm', 'LesionCHB', 'LesionUNC','Tissue'};

colors = {[0.75 0.5 0], [1 0.75 0], [0 0.75 1], [0 0.5 0.75], [0.5 0.75 0], [0 0 0], [0.5 0.5 0.5], [0.75 0 0.25]};

niceNameMap = containers.Map(keys,values);
niceColorMap = containers.Map(keys,colors);


