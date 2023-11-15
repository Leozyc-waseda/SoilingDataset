function tdata = train_model_combine_factors(ftdata,tdata,csize,conf)

%--------------------------------------------------------------------------
% Voting
tdata.COMBINE.predictData = ftdata{conf.feature_num + 1,2};
tdata.COMBINE.roundData   = ftdata{conf.feature_num + 2,2};

for i=1:size(tdata.COMBINE.roundData,1)
    vote = zeros(csize,1);
	for j=1:size(tdata.COMBINE.roundData,2)
        vote(tdata.COMBINE.roundData(i,j),:) = vote(tdata.COMBINE.roundData(i,j),:) + 1; 
    end
    
    maxVote = 0;
    for j=1:size(vote,1)
        if vote(j,:) > maxVote
            maxVote = vote(j,:);
            tdata.COMBINE.Vote(i,:) = j;
        end
    end
end

%--------------------------------------------------------------------------
% Max Certanty
tdata.COMBINE.CertData = abs(tdata.COMBINE.predictData - tdata.COMBINE.roundData);
for i=1:size(tdata.COMBINE.roundData,1)
    minCert = 1;
    for j=1:size(tdata.COMBINE.roundData,2)
        if tdata.COMBINE.CertData(i,j) < minCert
            minCert = tdata.COMBINE.CertData(i,j);
            tdata.COMBINE.Cert(i,:) = tdata.COMBINE.roundData(i,j);
        end
    end    
end

%--------------------------------------------------------------------------
% Average
tdata.COMBINE.Avg  = mean(tdata.COMBINE.predictData,2);
tdata.COMBINE.Ravg = round(tdata.COMBINE.Avg);