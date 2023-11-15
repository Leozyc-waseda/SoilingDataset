function vnum = matlabVersion()
vinfo = ver('matlab');
vstr = vinfo.Version;
vnum = str2num(vstr);
if strcmp(vstr,'7.10') % exception for R2010a
    vnum = 7.95;
end