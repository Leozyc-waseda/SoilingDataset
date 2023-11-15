% //////////////////////////////////////////////////////////////////// %
%           Surprise Linear Model - Copyright (C) 2004-2007            %
% by the University of Southern California (USC) and the iLab at USC.  %
% See http://iLab.usc.edu for information about this project.          %
% //////////////////////////////////////////////////////////////////// %
% This file is part of the iLab Neuromorphic Vision Toolkit            %
%                                                                      %
% The Surprise Linear Model is free software; you can                  %
% redistribute it and/or modify it under the terms of the GNU General  %
% Public License as published by the Free Software Foundation; either  %
% version 2 of the License, or (at your option) any later version.     %
%                                                                      %
% The Surprise Linear Model is distributed in the hope                 %
% that it will be useful, but WITHOUT ANY WARRANTY; without even the   %
% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      %
% PURPOSE.  See the GNU General Public License for more details.       %
%                                                                      %
% You should have received a copy of the GNU General Public License    %
% along with the iBaysian Surprise Matlab Toolkit; if not, write       %
% to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   %
% Boston, MA 02111-1307 USA.                                           %
% //////////////////////////////////////////////////////////////////// %
%
% Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
% $Revision: 55 $ 
% $Id$
% $HeadURL: https://surprise-mltk.svn.sourceforge.net/svnroot/surprise-mltk/source/surprise_toolkit/example_graph.m $

function conf = linear_model_set_up_channels(conf)

if strcmp(conf.condString,'UCIO_basic')         || strcmp(conf.condString,'NATHAN_UCIO_basic') || strcmp(conf.condString,'UCIO_legacy')      || ...
   strcmp(conf.condString,'JointGG_UCIO_basic') || strcmp(conf.condString,'UCIO_opt')          || strcmp(conf.condString,'JointGG_UCIO_opt') || ...
   strcmp(conf.condString,'NATHAN_UCIO_opt')    || strcmp(conf.condString,'UCIO_PoissonConst') 
    conf.feature_num = 10;
    conf.useFeature = 5;
    conf.feature_label{1}   = 'by';
    conf.feature_label{2}   = 'intensity';
    conf.feature_label{3}   = 'ori_0';
    conf.feature_label{4}   = 'ori_1';
    conf.feature_label{5}   = 'ori_2';
    conf.feature_label{6}   = 'ori_3';
    conf.feature_label{7}   = 'rg';
    conf.feature_label{8}   = 'final';
    conf.feature_label{9}   = 'final-lam';
    conf.feature_label{10}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIO_basic')              || strcmp(conf.condString,'UHIO_legacy')           || strcmp(conf.condString,'UHIO_opt')            || ...
       strcmp(conf.condString,'JointGG_UHIO_basic')      || strcmp(conf.condString,'JointGG_UHIO_opt')      || strcmp(conf.condString,'Gaussian_UHIO_basic') || ...
       strcmp(conf.condString,'Outlier_UHIO_basic')      || strcmp(conf.condString,'Outlier_UHIO_opt')      || strcmp(conf.condString,'UHIO_train')          || ...
       strcmp(conf.condString,'ChiSq_UHIO_basic')        || strcmp(conf.condString,'ChiSq_UHIO_opt')        || strcmp(conf.condString,'UHIO_max')            || ...
       strcmp(conf.condString,'PoissonConst_UHIO_basic') || strcmp(conf.condString,'PoissonConst_UHIO_opt') 
    conf.feature_num = 12;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'final';  
    conf.feature_label{11}  = 'final-lam';
    conf.feature_label{12}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHO_basic')              || strcmp(conf.condString,'UHO_legacy')           || strcmp(conf.condString,'UHO_opt')            || ...
       strcmp(conf.condString,'JointGG_UHO_basic')      || strcmp(conf.condString,'JointGG_UHO_opt')      || strcmp(conf.condString,'Gaussian_UHO_basic') || ...
       strcmp(conf.condString,'Outlier_UHO_basic')      || strcmp(conf.condString,'Outlier_UHO_opt')      || strcmp(conf.condString,'UHO_train')          || ...
       strcmp(conf.condString,'ChiSq_UHO_basic')        || strcmp(conf.condString,'ChiSq_UHO_opt')        || strcmp(conf.condString,'UHO_max')            || ...
       strcmp(conf.condString,'PoissonConst_UHO_basic') || strcmp(conf.condString,'PoissonConst_UHO_opt')
    conf.feature_num = 11;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'final';    
    conf.feature_label{10}  = 'final-lam';
    conf.feature_label{11}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOLTWX_basic') || strcmp(conf.condString,'UHOLTX_legacy') || strcmp(conf.condString,'UHOLTWX_opt')
    conf.feature_num = 30;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}  = 'junction_10101010';
    conf.feature_label{10}  = 'junction_10100000';
    conf.feature_label{11}  = 'junction_00101000';    
    conf.feature_label{12}  = 'junction_00001010';
    conf.feature_label{13}  = 'junction_10000010';
    conf.feature_label{14}  = 'junction_10001010';
    conf.feature_label{15}  = 'junction_10100010';    
    conf.feature_label{16}  = 'junction_10101000';
    conf.feature_label{17}  = 'junction_00101010';  
    conf.feature_label{18}  = 'junction_01010101';
    conf.feature_label{19}  = 'junction_01010000';
    conf.feature_label{20}  = 'junction_00010100';    
    conf.feature_label{21}  = 'junction_00000101';
    conf.feature_label{22}  = 'junction_01000001';
    conf.feature_label{23}  = 'junction_01000101';
    conf.feature_label{24}  = 'junction_01010001';    
    conf.feature_label{25}  = 'junction_01010100';
    conf.feature_label{26}  = 'junction_00010101';
    conf.feature_label{27}  = 'contour';
    conf.feature_label{28}  = 'final';  
    conf.feature_label{29}  = 'final-lam';
    conf.feature_label{30}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOLTWX_basic') || strcmp(conf.condString,'UHIOLTX_legacy') || strcmp(conf.condString,'UHIOLTWX_opt')
    conf.feature_num = 31;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10101010';
    conf.feature_label{11}  = 'junction_10100000';
    conf.feature_label{12}  = 'junction_00101000';    
    conf.feature_label{13}  = 'junction_00001010';
    conf.feature_label{14}  = 'junction_10000010';
    conf.feature_label{15}  = 'junction_10001010';
    conf.feature_label{16}  = 'junction_10100010';    
    conf.feature_label{17}  = 'junction_10101000';
    conf.feature_label{18}  = 'junction_00101010';  
    conf.feature_label{19}  = 'junction_01010101';
    conf.feature_label{20}  = 'junction_01010000';
    conf.feature_label{21}  = 'junction_00010100';    
    conf.feature_label{22}  = 'junction_00000101';
    conf.feature_label{23}  = 'junction_01000001';
    conf.feature_label{24}  = 'junction_01000101';
    conf.feature_label{25}  = 'junction_01010001';    
    conf.feature_label{26}  = 'junction_01010100';
    conf.feature_label{27}  = 'junction_00010101';
    conf.feature_label{28}  = 'contour';
    conf.feature_label{29}  = 'final'; 
    conf.feature_label{30}  = 'final-lam';
    conf.feature_label{31}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOLTWXE_basic') || strcmp(conf.condString,'UHIOLTWXE_opt') || strcmp(conf.condString,'UHIOLTWXE_train')
    conf.feature_num = 39;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10101010';
    conf.feature_label{11}  = 'junction_10100000';
    conf.feature_label{12}  = 'junction_00101000';    
    conf.feature_label{13}  = 'junction_00001010';
    conf.feature_label{14}  = 'junction_10000010';
    conf.feature_label{15}  = 'junction_10001010';
    conf.feature_label{16}  = 'junction_10100010';    
    conf.feature_label{17}  = 'junction_10101000';       
    conf.feature_label{18}  = 'junction_00101010';    
    conf.feature_label{19}  = 'junction_10000000';
    conf.feature_label{20}  = 'junction_00100000';
    conf.feature_label{21}  = 'junction_00001000';
    conf.feature_label{22}  = 'junction_00000010'; 
    conf.feature_label{23}  = 'junction_01010101';
    conf.feature_label{24}  = 'junction_01010000';
    conf.feature_label{25}  = 'junction_00010100';    
    conf.feature_label{26}  = 'junction_00000101';
    conf.feature_label{27}  = 'junction_01000001';
    conf.feature_label{28}  = 'junction_01000101';
    conf.feature_label{29}  = 'junction_01010001';    
    conf.feature_label{30}  = 'junction_01010100';
    conf.feature_label{31}  = 'junction_00010101';    
    conf.feature_label{32}  = 'junction_01000000';
    conf.feature_label{33}  = 'junction_00010000';
    conf.feature_label{34}  = 'junction_00000100';
    conf.feature_label{35}  = 'junction_00000001';
    conf.feature_label{36}  = 'contour';
    conf.feature_label{37}  = 'final'; 
    conf.feature_label{38}  = 'final-lam';
    conf.feature_label{39}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOLTXE_basic')              || strcmp(conf.condString,'UHIOLTXE_opt')              || ...
       strcmp(conf.condString,'JointGG_UHIOLTXE_basic')      || strcmp(conf.condString,'JointGG_UHIOLTXE_opt')      || ...
       strcmp(conf.condString,'ChiSq_UHIOLTXE_basic')        || strcmp(conf.condString,'ChiSq_UHIOLTXE_opt')        || ...
       strcmp(conf.condString,'UHIOLTXE_train')              || strcmp(conf.condString,'UHIOLTXE_max')              || ...
       strcmp(conf.condString,'PoissonConst_UHIOLTXE_basic') || strcmp(conf.condString,'PoissonConst_UHIOLTXE_opt') 
    conf.feature_num = 38;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10101010';
    conf.feature_label{11}  = 'junction_10100000';
    conf.feature_label{12}  = 'junction_00101000';    
    conf.feature_label{13}  = 'junction_00001010';
    conf.feature_label{14}  = 'junction_10000010';
    conf.feature_label{15}  = 'junction_10001010';
    conf.feature_label{16}  = 'junction_10100010';    
    conf.feature_label{17}  = 'junction_10101000';
    conf.feature_label{18}  = 'junction_00101010';    
    conf.feature_label{19}  = 'junction_10000000';
    conf.feature_label{20}  = 'junction_00100000';
    conf.feature_label{21}  = 'junction_00001000';
    conf.feature_label{22}  = 'junction_00000010'; 
    conf.feature_label{23}  = 'junction_01010101';
    conf.feature_label{24}  = 'junction_01010000';
    conf.feature_label{25}  = 'junction_00010100';    
    conf.feature_label{26}  = 'junction_00000101';
    conf.feature_label{27}  = 'junction_01000001';
    conf.feature_label{28}  = 'junction_01000101';
    conf.feature_label{29}  = 'junction_01010001';    
    conf.feature_label{30}  = 'junction_01010100';
    conf.feature_label{31}  = 'junction_00010101';    
    conf.feature_label{32}  = 'junction_01000000';
    conf.feature_label{33}  = 'junction_00010000';
    conf.feature_label{34}  = 'junction_00000100';
    conf.feature_label{35}  = 'junction_00000001';
    conf.feature_label{36}  = 'final';  
    conf.feature_label{37}  = 'final-lam';
    conf.feature_label{38}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOLTXE_basic')                || strcmp(conf.condString,'UHOLTXE_opt')              || ...
       strcmp(conf.condString,'JointGG_UHOLTXE_basic')        || strcmp(conf.condString,'JointGG_UHOLTXE_opt')      || ...
       strcmp(conf.condString,'ChiSq_UHOLTXE_basic')          || strcmp(conf.condString,'ChiSq_UHOLTXE_opt')        || ...
       strcmp(conf.condString,'UHOLTXE_train')                || strcmp(conf.condString,'UHOLTXE_max')              || ...
       strcmp(conf.condString,'Outlier_UHOLTXE_basic')        || strcmp(conf.condString,'Outlier_UHOLTXE_opt')      || ...
       strcmp(conf.condString,'Gaussian_UHOLTXE_basic')       || strcmp(conf.condString,'Gaussian_UHOLTXE_opt')     || ...
       strcmp(conf.condString,'PoissonConst_UHOLTXE_basic')   || strcmp(conf.condString,'PoissonConst_UHOLTXE_opt') || ...
       strcmp(conf.condString,'PoissonFloat_UHOLTXE_basic')   || strcmp(conf.condString,'PoissonFloat_UHOLTXE_opt') 
    conf.feature_num = 37;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'junction_10101010';
    conf.feature_label{10}  = 'junction_10100000';
    conf.feature_label{11}  = 'junction_00101000';    
    conf.feature_label{12}  = 'junction_00001010';
    conf.feature_label{13}  = 'junction_10000010';
    conf.feature_label{14}  = 'junction_10001010';
    conf.feature_label{15}  = 'junction_10100010';    
    conf.feature_label{16}  = 'junction_10101000';
    conf.feature_label{17}  = 'junction_00101010';    
    conf.feature_label{18}  = 'junction_10000000';
    conf.feature_label{19}  = 'junction_00100000';
    conf.feature_label{20}  = 'junction_00001000';
    conf.feature_label{21}  = 'junction_00000010'; 
    conf.feature_label{22}  = 'junction_01010101';
    conf.feature_label{23}  = 'junction_01010000';
    conf.feature_label{24}  = 'junction_00010100';    
    conf.feature_label{25}  = 'junction_00000101';
    conf.feature_label{26}  = 'junction_01000001';
    conf.feature_label{27}  = 'junction_01000101';
    conf.feature_label{28}  = 'junction_01010001';    
    conf.feature_label{29}  = 'junction_01010100';
    conf.feature_label{30}  = 'junction_00010101';    
    conf.feature_label{31}  = 'junction_01000000';
    conf.feature_label{32}  = 'junction_00010000';
    conf.feature_label{33}  = 'junction_00000100';
    conf.feature_label{34}  = 'junction_00000001';
    conf.feature_label{35}  = 'final';
    conf.feature_label{36}  = 'final-lam';
    conf.feature_label{37}  = 'final-AGmask';
elseif strcmp(conf.condString,'UQOLTXE_basic')              || strcmp(conf.condString,'UQOLTXE_opt')              || ...
       strcmp(conf.condString,'JointGG_UQOLTXE_basic')      || strcmp(conf.condString,'JointGG_UQOLTXE_opt')      || ...
       strcmp(conf.condString,'ChiSq_UQOLTXE_basic')        || strcmp(conf.condString,'ChiSq_UQOLTXE_opt')        || ...
       strcmp(conf.condString,'UQOLTXE_train')              || strcmp(conf.condString,'UQOLTXE_max')              || ...
       strcmp(conf.condString,'PoissonConst_UQOLTXE_basic') || strcmp(conf.condString,'PoissonConst_UQOLTXE_opt') 
    conf.feature_num = 36;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'val';
    conf.feature_label{8}   = 'junction_10101010';
    conf.feature_label{9}   = 'junction_10100000';
    conf.feature_label{10}  = 'junction_00101000';    
    conf.feature_label{11}  = 'junction_00001010';
    conf.feature_label{12}  = 'junction_10000010';
    conf.feature_label{13}  = 'junction_10001010';
    conf.feature_label{14}  = 'junction_10100010';    
    conf.feature_label{15}  = 'junction_10101000';
    conf.feature_label{16}  = 'junction_00101010';    
    conf.feature_label{17}  = 'junction_10000000';
    conf.feature_label{18}  = 'junction_00100000';
    conf.feature_label{19}  = 'junction_00001000';
    conf.feature_label{20}  = 'junction_00000010'; 
    conf.feature_label{21}  = 'junction_01010101';
    conf.feature_label{22}  = 'junction_01010000';
    conf.feature_label{23}  = 'junction_00010100';    
    conf.feature_label{24}  = 'junction_00000101';
    conf.feature_label{25}  = 'junction_01000001';
    conf.feature_label{26}  = 'junction_01000101';
    conf.feature_label{27}  = 'junction_01010001';    
    conf.feature_label{28}  = 'junction_01010100';
    conf.feature_label{29}  = 'junction_00010101';    
    conf.feature_label{30}  = 'junction_01000000';
    conf.feature_label{31}  = 'junction_00010000';
    conf.feature_label{32}  = 'junction_00000100';
    conf.feature_label{33}  = 'junction_00000001';
    conf.feature_label{34}  = 'final';    
    conf.feature_label{35}  = 'final-lam';
    conf.feature_label{36}  = 'final-AGmask';
elseif strcmp(conf.condString,'UCIOLTXE_basic')              || strcmp(conf.condString,'UCIOLTXE_opt')              || ...
       strcmp(conf.condString,'JointGG_UCIOLTXE_basic')      || strcmp(conf.condString,'JointGG_UCIOLTXE_opt')      || ...
       strcmp(conf.condString,'ChiSq_UCIOLTXE_basic')        || strcmp(conf.condString,'ChiSq_UCIOLTXE_opt')        || ...
       strcmp(conf.condString,'UCIOLTXE_train')              || strcmp(conf.condString,'UCIOLTXE_max')              || ...
       strcmp(conf.condString,'PoissonConst_UCIOLTXE_basic') || strcmp(conf.condString,'PoissonConst_UCIOLTXE_opt') 
    conf.feature_num = 36;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'rg';
    conf.feature_label{7}   = 'by';
    conf.feature_label{8}   = 'junction_10101010';
    conf.feature_label{9}   = 'junction_10100000';
    conf.feature_label{10}  = 'junction_00101000';    
    conf.feature_label{11}  = 'junction_00001010';
    conf.feature_label{12}  = 'junction_10000010';
    conf.feature_label{13}  = 'junction_10001010';
    conf.feature_label{14}  = 'junction_10100010';    
    conf.feature_label{15}  = 'junction_10101000';
    conf.feature_label{16}  = 'junction_00101010';    
    conf.feature_label{17}  = 'junction_10000000';
    conf.feature_label{18}  = 'junction_00100000';
    conf.feature_label{19}  = 'junction_00001000';
    conf.feature_label{20}  = 'junction_00000010'; 
    conf.feature_label{21}  = 'junction_01010101';
    conf.feature_label{22}  = 'junction_01010000';
    conf.feature_label{23}  = 'junction_00010100';    
    conf.feature_label{24}  = 'junction_00000101';
    conf.feature_label{25}  = 'junction_01000001';
    conf.feature_label{26}  = 'junction_01000101';
    conf.feature_label{27}  = 'junction_01010001';    
    conf.feature_label{28}  = 'junction_01010100';
    conf.feature_label{29}  = 'junction_00010101';    
    conf.feature_label{30}  = 'junction_01000000';
    conf.feature_label{31}  = 'junction_00010000';
    conf.feature_label{32}  = 'junction_00000100';
    conf.feature_label{33}  = 'junction_00000001';
    conf.feature_label{34}  = 'final'; 
    conf.feature_label{35}  = 'final-lam';
    conf.feature_label{36}  = 'final-AGmask';
elseif strcmp(conf.condString,'UIOLTXE_basic')              || strcmp(conf.condString,'UIOLTXE_opt')              || ...
       strcmp(conf.condString,'JointGG_UIOLTXE_basic')      || strcmp(conf.condString,'JointGG_UIOLTXE_opt')      || ...
       strcmp(conf.condString,'ChiSq_UIOLTXE_basic')        || strcmp(conf.condString,'ChiSq_UIOLTXE_opt')        || ...
       strcmp(conf.condString,'UIOLTXE_train')              || strcmp(conf.condString,'UIOLTXE_max')              || ...
       strcmp(conf.condString,'PoissonConst_UIOLTXE_basic') || strcmp(conf.condString,'PoissonConst_UIOLTXE_opt') 
    conf.feature_num = 34;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'junction_10101010';
    conf.feature_label{7}   = 'junction_10100000';
    conf.feature_label{8}  = 'junction_00101000';    
    conf.feature_label{9}  = 'junction_00001010';
    conf.feature_label{10}  = 'junction_10000010';
    conf.feature_label{11}  = 'junction_10001010';
    conf.feature_label{12}  = 'junction_10100010';    
    conf.feature_label{13}  = 'junction_10101000';
    conf.feature_label{14}  = 'junction_00101010';    
    conf.feature_label{15}  = 'junction_10000000';
    conf.feature_label{16}  = 'junction_00100000';
    conf.feature_label{17}  = 'junction_00001000';
    conf.feature_label{18}  = 'junction_00000010'; 
    conf.feature_label{19}  = 'junction_01010101';
    conf.feature_label{20}  = 'junction_01010000';
    conf.feature_label{21}  = 'junction_00010100';    
    conf.feature_label{22}  = 'junction_00000101';
    conf.feature_label{23}  = 'junction_01000001';
    conf.feature_label{24}  = 'junction_01000101';
    conf.feature_label{25}  = 'junction_01010001';    
    conf.feature_label{26}  = 'junction_01010100';
    conf.feature_label{27}  = 'junction_00010101';    
    conf.feature_label{28}  = 'junction_01000000';
    conf.feature_label{29}  = 'junction_00010000';
    conf.feature_label{30}  = 'junction_00000100';
    conf.feature_label{31}  = 'junction_00000001';
    conf.feature_label{32}  = 'final'; 
    conf.feature_label{33}  = 'final-lam';
    conf.feature_label{34}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOE_basic')         || strcmp(conf.condString,'UHIOE_opt')         || ...
       strcmp(conf.condString,'JointGG_UHIOE_basic') || strcmp(conf.condString,'JointGG_UHIOE_opt') || ...
       strcmp(conf.condString,'UHIOE_train') 
    conf.feature_num = 20;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10000000';
    conf.feature_label{11}  = 'junction_00100000';
    conf.feature_label{12}  = 'junction_00001000';
    conf.feature_label{13}  = 'junction_00000010'; 
    conf.feature_label{14}  = 'junction_01000000';
    conf.feature_label{15}  = 'junction_00010000';
    conf.feature_label{16}  = 'junction_00000100';
    conf.feature_label{17}  = 'junction_00000001';
    conf.feature_label{18}  = 'final';    
    conf.feature_label{19}  = 'final-lam';
    conf.feature_label{20}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOL_basic')         || strcmp(conf.condString,'UHIOL_opt')         || ...
       strcmp(conf.condString,'JointGG_UHIOL_basic') || strcmp(conf.condString,'JointGG_UHIOL_opt') || ...
       strcmp(conf.condString,'UHIOL_train') 
    conf.feature_num = 20;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10100000';
    conf.feature_label{11}  = 'junction_00101000';    
    conf.feature_label{12}  = 'junction_00001010';
    conf.feature_label{13}  = 'junction_10000010'; 
    conf.feature_label{14}  = 'junction_01010000';
    conf.feature_label{15}  = 'junction_00010100';    
    conf.feature_label{16}  = 'junction_00000101';
    conf.feature_label{17}  = 'junction_01000001';  
    conf.feature_label{18}  = 'final';    
    conf.feature_label{19}  = 'final-lam';
    conf.feature_label{20}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOT_basic')         || strcmp(conf.condString,'UHIOT_opt')         || ...
       strcmp(conf.condString,'JointGG_UHIOT_basic') || strcmp(conf.condString,'JointGG_UHIOT_opt') || ...
       strcmp(conf.condString,'UHIOT_train') 
    conf.feature_num = 20;
    conf.useFeature = 4;
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10001010';
    conf.feature_label{11}  = 'junction_10100010';    
    conf.feature_label{12}  = 'junction_10101000';
    conf.feature_label{13}  = 'junction_00101010'; 
    conf.feature_label{14}  = 'junction_01000101';
    conf.feature_label{15}  = 'junction_01010001';    
    conf.feature_label{16}  = 'junction_01010100';
    conf.feature_label{17}  = 'junction_00010101';     
    conf.feature_label{18}  = 'final';    
    conf.feature_label{19}  = 'final-lam';
    conf.feature_label{20}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOX_basic')         || strcmp(conf.condString,'UHIOX_opt')         || ...
       strcmp(conf.condString,'JointGG_UHIOX_basic') || strcmp(conf.condString,'JointGG_UHIOX_opt') || ...
       strcmp(conf.condString,'UHIOX_train') 
    conf.feature_num = 15;
    conf.useFeature = 4;   
    conf.feature_label{1}   = 'intensity';
    conf.feature_label{2}   = 'ori_0';
    conf.feature_label{3}   = 'ori_1';
    conf.feature_label{4}   = 'ori_2';
    conf.feature_label{5}   = 'ori_3';
    conf.feature_label{6}   = 'h1';
    conf.feature_label{7}   = 'h2';
    conf.feature_label{8}   = 'sat';
    conf.feature_label{9}   = 'val';
    conf.feature_label{10}  = 'junction_10101010';
    conf.feature_label{12}  = 'junction_01010101';
    conf.feature_label{13}  = 'final';  
    conf.feature_label{14}  = 'final-lam';
    conf.feature_label{15}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOE_basic')         || strcmp(conf.condString,'UHOE_opt')         || ...
       strcmp(conf.condString,'JointGG_UHOE_basic') || strcmp(conf.condString,'JointGG_UHOE_opt') || ...
       strcmp(conf.condString,'UHOE_train') 
    conf.feature_num = 19;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'junction_10000000';
    conf.feature_label{10}  = 'junction_00100000';
    conf.feature_label{11}  = 'junction_00001000';
    conf.feature_label{12}  = 'junction_00000010'; 
    conf.feature_label{13}  = 'junction_01000000';
    conf.feature_label{14}  = 'junction_00010000';
    conf.feature_label{15}  = 'junction_00000100';
    conf.feature_label{16}  = 'junction_00000001';
    conf.feature_label{17}  = 'final';    
    conf.feature_label{18}  = 'final-lam';
    conf.feature_label{19}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOL_basic')         || strcmp(conf.condString,'UHOL_opt')         || ...
       strcmp(conf.condString,'JointGG_UHOL_basic') || strcmp(conf.condString,'JointGG_UHOL_opt') || ...
       strcmp(conf.condString,'UHOL_train') 
    conf.feature_num = 19;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'junction_10100000';
    conf.feature_label{10}  = 'junction_00101000';    
    conf.feature_label{11}  = 'junction_00001010';
    conf.feature_label{12}  = 'junction_10000010'; 
    conf.feature_label{13}  = 'junction_01010000';
    conf.feature_label{14}  = 'junction_00010100';    
    conf.feature_label{15}  = 'junction_00000101';
    conf.feature_label{16}  = 'junction_01000001';  
    conf.feature_label{17}  = 'final';    
    conf.feature_label{18}  = 'final-lam';
    conf.feature_label{19}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOT_basic')         || strcmp(conf.condString,'UHOT_opt')         || ...
       strcmp(conf.condString,'JointGG_UHOT_basic') || strcmp(conf.condString,'JointGG_UHOT_opt') || ...
       strcmp(conf.condString,'UHOT_train') 
    conf.feature_num = 19;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'junction_10001010';
    conf.feature_label{10}  = 'junction_10100010';    
    conf.feature_label{11}  = 'junction_10101000';
    conf.feature_label{12}  = 'junction_00101010'; 
    conf.feature_label{13}  = 'junction_01000101';
    conf.feature_label{14}  = 'junction_01010001';    
    conf.feature_label{15}  = 'junction_01010100';
    conf.feature_label{16}  = 'junction_00010101';     
    conf.feature_label{17}  = 'final';    
    conf.feature_label{18}  = 'final-lam';
    conf.feature_label{19}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOX_basic')         || strcmp(conf.condString,'UHOX_opt')         || ...
       strcmp(conf.condString,'JointGG_UHOX_basic') || strcmp(conf.condString,'JointGG_UHOX_opt') || ...
       strcmp(conf.condString,'UHOX_train') 
    conf.feature_num = 13;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'junction_10101010';
    conf.feature_label{10}  = 'junction_01010101';
    conf.feature_label{11}  = 'final';    
    conf.feature_label{12}  = 'final-lam';
    conf.feature_label{13}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHOW_basic')         || strcmp(conf.condString,'UHOW_opt')         || ...
       strcmp(conf.condString,'JointGG_UHOW_basic') || strcmp(conf.condString,'JointGG_UHOW_opt') || ...
       strcmp(conf.condString,'UHOW_train') 
    conf.feature_num = 13;
    conf.useFeature = 3;
    conf.feature_label{1}   = 'ori_0';
    conf.feature_label{2}   = 'ori_1';
    conf.feature_label{3}   = 'ori_2';
    conf.feature_label{4}   = 'ori_3';
    conf.feature_label{5}   = 'h1';
    conf.feature_label{6}   = 'h2';
    conf.feature_label{7}   = 'sat';
    conf.feature_label{8}   = 'val';
    conf.feature_label{9}   = 'contour';
    conf.feature_label{11}  = 'final';    
    conf.feature_label{12}  = 'final-lam';
    conf.feature_label{13}  = 'final-AGmask';
elseif strcmp(conf.condString,'UHIOGKSE_legacy') 
    conf.feature_num = 28;
    conf.useFeature = 4;
    conf.feature_label{1}    = 'intensity';
    conf.feature_label{2}    = 'ori_0';
    conf.feature_label{3}    = 'ori_1';
    conf.feature_label{4}    = 'ori_2';
    conf.feature_label{5}    = 'ori_3';
    conf.feature_label{6}    = 'h1';
    conf.feature_label{7}    = 'h2';
    conf.feature_label{8}    = 'sat';
    conf.feature_label{9}    = 'val';
    conf.feature_label{10}   = 'rg';
    conf.feature_label{11}   = 'by';
    conf.feature_label{12}   = 'r-g';
    conf.feature_label{13}   = 'g-r';
    conf.feature_label{14}   = 'b-y';
    conf.feature_label{15}   = 'y-b';
    conf.feature_label{16}   = 'cband_0';
    conf.feature_label{17}   = 'cband_1';
    conf.feature_label{18}   = 'cband_2';
    conf.feature_label{19}   = 'cband_3';
    conf.feature_label{20}   = 'cband_4';
    conf.feature_label{21}   = 'cband_5';
    conf.feature_label{22}   = 'junction_10000000';
    conf.feature_label{23}   = 'junction_00100000';
    conf.feature_label{24}   = 'junction_00001000';
    conf.feature_label{25}   = 'junction_00000010';
    conf.feature_label{26}   = 'final';    
    conf.feature_label{27}   = 'final-lam';
    conf.feature_label{28}   = 'final-AGmask';
elseif strcmp(conf.condString,'UCIO_old') 
    conf.feature_num = 15;
    conf.useFeature = 5;
    conf.feature_label{1}   = 'by';
    conf.feature_label{2}   = 'intensity';
    conf.feature_label{3}   = 'ori_0';
    conf.feature_label{4}   = 'ori_1';
    conf.feature_label{5}   = 'ori_2';
    conf.feature_label{6}   = 'ori_3';
    conf.feature_label{7}   = 'rg';  
    conf.feature_label{8}   = 'dir_0';
    conf.feature_label{9}   = 'dir_1';
    conf.feature_label{10}  = 'dir_2';
    conf.feature_label{11}  = 'dir_3';
    conf.feature_label{12}  = 'flicker';
    conf.feature_label{13}  = 'final';       
    conf.feature_label{14}  = 'final-lam';
    conf.feature_label{15}  = 'final-AGmask';
else
    error('Condition string \"conf.condString\" set as \"%s\" is not valid',conf.condString);
end