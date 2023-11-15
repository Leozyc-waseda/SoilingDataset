% sort by offset and by stim ABtype (Anims-Trans v Trans-Anims)

function [ldata_AB,conf] = linear_model_frame_offset_sort_AB(ldata_AB,conf)

dprint(['Getting AB Offsets']); 
ldata_AB.OFFSET_CELLS = max(max(ldata_AB.TFRAME_DIFF));

% create n new offset classes
for i=1:ldata_AB.OFFSET_CELLS
    ldata_AB.OFFCLASS{i}.N = 0;
end

% how many samples per bin
for i=1:size(ldata_AB.SAMPLE,1)
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.N = ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.N + 1;
end
    
% Resize containers
for i=1:ldata_AB.OFFSET_CELLS
    ldata_AB.OFFCLASS{i}.TFRAME_A    = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.TFRAME_B    = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.TFRAME_DIFF = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.CLASS       = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.ABType      = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.MAX         = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.MIN         = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.AVG         = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.STD         = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.MAXX        = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.MAXY        = zeros(ldata_AB.OFFCLASS{i}.N,1);
%    ldata_AB.OFFCLASS{i}.FEATURE     = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.OLDFRAME    = zeros(ldata_AB.OFFCLASS{i}.N,1);
    ldata_AB.OFFCLASS{i}.NEWFRAME    = zeros(ldata_AB.OFFCLASS{i}.N,1);
end

% Reset N to reuse
for i=1:ldata_AB.OFFSET_CELLS
    ldata_AB.OFFCLASS{i}.N = 0;
end

% sort each sample into an offset class
for i=1:size(ldata_AB.SAMPLE,1)
    n = ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.N + 1;
    
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.TFRAME_A(n,1)    = ldata_AB.TFRAME_A(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.TFRAME_B(n,1)    = ldata_AB.TFRAME_B(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.TFRAME_DIFF(n,1) = ldata_AB.TFRAME_DIFF(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.CLASS(n,1)       = ldata_AB.CLASS(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.ABType(n,1)      = ldata_AB.ABType(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.MAX(n,1)         = ldata_AB.MAX(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.MIN(n,1)         = ldata_AB.MIN(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.AVG(n,1)         = ldata_AB.AVG(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.STD(n,1)         = ldata_AB.STD(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.MAXX(n,1)        = ldata_AB.MAXX(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.MAXY(n,1)        = ldata_AB.MAXY(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.FEATURE{n,1}     = ldata_AB.FEATURE{i,1};
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.OLDFRAME(n,1)    = ldata_AB.OLDFRAME(i,1);
    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.NEWFRAME(n,1)    = ldata_AB.NEWFRAME(i,1);

    ldata_AB.OFFCLASS{ldata_AB.TFRAME_DIFF(i,1)}.N = n;
end
dprint(['Done']);