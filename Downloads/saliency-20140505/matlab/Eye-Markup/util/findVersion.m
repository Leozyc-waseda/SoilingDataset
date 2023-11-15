function vnum = findVersion()
vinfo = ver('matlab');
vnum = str2num(vinfo.Version);