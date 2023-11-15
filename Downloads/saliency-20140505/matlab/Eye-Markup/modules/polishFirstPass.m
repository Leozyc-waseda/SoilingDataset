function result = polishFirstPass(data,args)
scode = getvalue('code', args);
sc_list = [scode.SACCADE scode.SAC_CMBND ...
    scode.BLINK scode.SMOOTH ...
    scode.FIXATION];
result = cleanFinalTrace(data,sc_list,args);
end