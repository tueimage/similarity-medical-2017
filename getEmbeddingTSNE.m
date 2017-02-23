function bestSolution = getEmbeddingTSNE(data, perplexity, numTries)

bestSolution = [];
bestCost = inf;


for i=1:numTries
    [tSup, tCost]=tsne(data,[],[],perplexity);
    if tCost < bestCost
        bestSolution = tSup;
        bestCost = tCost;
    end
end