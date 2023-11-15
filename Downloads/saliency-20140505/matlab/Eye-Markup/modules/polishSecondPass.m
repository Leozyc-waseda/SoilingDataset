function result = polishSecondPass(data,args)
scode = getvalue('code', args);
sc_list = [scode.BLINK scode.SMOOTH ...
    scode.FIXATION];

result = cleanFinalTrace(data,sc_list,args);
end
