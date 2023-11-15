function data = fishyates(data)
%will shuffle the rows of data
for (ii = size(data,1):-1:2)
    r = floor(unifrnd(1,ii+1));
    temp = data(r,:);
    data(r,:) = data(ii,:);
    data(ii,:) = temp;
end