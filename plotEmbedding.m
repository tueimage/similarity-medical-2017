function legendNames = plotEmbedding(tsneRank, metaLabelNameAll, niceNameMap, niceColorMap, legendFlag)

tsneRank = prdataset(tsneRank, metaLabelNameAll);

ml = unique(metaLabelNameAll, 'rows');

for i=1:size(ml,1)
    c = seldat(tsneRank, ml(i,:)); 
    try
        legendNames{i} = niceNameMap(strtrim(ml(i,:)));
    catch
        keyboard
    end
    thisColor = niceColorMap(strtrim(ml(i,:)));
    scatter(+c(:,1), +c(:,2), 'MarkerEdgeColor', thisColor, 'MarkerFaceColor', thisColor);
    hold on;
end

%set(gca,'FontName','Georgia','FontSize',12);
set(gca,'Color',[1 1 1])
set(gca, 'XTickLabel', '');
set(gca, 'YTickLabel', '');

if nargin==5 && ( (ischar(legendFlag) && strcmp(legendFlag, 'legend')) || (isnum(legendFlag) && legendFlag == 1))
    legend(legendNames,'interpreter','latex','location','NorthEast');
end