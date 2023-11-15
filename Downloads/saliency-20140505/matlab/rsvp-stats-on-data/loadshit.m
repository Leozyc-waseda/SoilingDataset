fid = fopen('Export.Results.Surprise.txt','W'); 

for i=1:75
    filename = ['\\magnarama\home\matlab.log\results.',num2str(i),'.mat'];
    SET = load(filename);
    Set{i} = SET.RESULTS.ValidBarSet;
    for x=1:size(Set{i},1)
        for y=1:size(Set{i},2)
            fprintf(fid,'%d\t',Set{i}(x,y));
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'\n');
end

fclose(fid);