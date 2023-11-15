function list = aliasList
% ALIASLIST returns a list of the aliases, as a cell array
% list = ALIASLIST 
%   Returns the list of all the alias definitions.  
%   The list is in the form of a Nx2 cell array, with 
%   the 1st column as the name of the alias,
%   and the 2nd column as the sub-parameters that would normally run under
%   markEye.
%   NB: Aliases cannot contain other aliases.

settings = defaultparams;
list = {
    'kingston_monkey',...  %kingston setup, monkey using a search coil
    {'sf',1000,'ppd',[10.39 9.99],'screen_size',[640 480]}; ...
    'kingston_human',...   %kingston setup, human with head mounted
    {'sf',250,'ppd',[19 19],'screen_size',[640 480], ...
                 'maxsaclength', 800}; ...
    'nips_monkey',...      %NIPS setup, monkey with search coil
    {'sf',1000,'ppd',[9.2 9.2],'screen_size',[640 480]}; ...
    'usc_low_res',...      %USC low res (640,480) setup 
    {'ppd',[11.7 14.7],'screen_size',[640 480]}; ...
    'strict',...           %strict thresholds
    {'pro_anglethresh',15,'pro_timethresh',75,...
                 'sac_mintime',25}; ...
    'loose',...            %more relaxed threshold
    {'pro_anglethresh',90,'pro_timethresh',160,...
                 'sac_mintime',15}; ...
    'strict_time',...      %strict time thresholds
    {'pro_timethresh',75}; ...
    'loose_time',...       %relaxed time thresholds
    {'pro_timethresh',160}; ...
    'strict_angle',...     %strict angle threshold
    {'pro_anglethresh',15}; ...
    'loose_angle', ...     %relaxed angle thresholds
    {'pro_anglethresh',90}; ...
    'debug', {'verbose', Inf};... % show all output
    'show_trace', {'verbose', 1};... % show trace on screen
    'quiet', {'verbose', -1};... % show minimal output
    'silent', {'verbose', -Inf};... % show no output
    'dual_core', {'ncores', 2}; ...
    'quad_core', {'ncores', 4}; ...
    'add_region',... % add region to the list of stats to be outputted
    {'stats', [getvalue('stats',settings), 'region']}...
};    
end