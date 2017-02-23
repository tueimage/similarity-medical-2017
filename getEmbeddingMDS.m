function mdsData = getEmbeddingMDS(allMetaData)

%MDS takes distances, set diagonal to 0 so MDS doesn't complain
distMetaData = sqeucldistm(+allMetaData,+allMetaData);
distMetaData(eye(size(distMetaData))==1) = 0;
mdsData = distMetaData*mds(distMetaData);