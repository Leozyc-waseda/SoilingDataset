% get auto, triangle and coke results, and plot them

clear all; close all;
cd /cit/imaging/saliency/STIMautobahn/DBtraining;
result;
data_a1 = data; avg_a1 = avg_info; max_a1 = max_info; min_a1 = min_info;
cd ../DBtest;
result;
data_a2 = data; avg_a2 = avg_info; max_a2 = max_info; min_a2 = min_info;
data_auto = [data_a1; data_a2]; avg_auto = [avg_a1 avg_a2];
max_auto = [max_a1 max_a2]; min_auto = [min_a1 min_a2];

cd /cit/imaging/saliency/STIMcoke/DBtraining;
result;
data_c1 = data; avg_c1 = avg_info; max_c1 = max_info; min_c1 = min_info;
cd ../DBtest;
result;
data_c2 = data; avg_c2 = avg_info; max_c2 = max_info; min_c2 = min_info;
data_coke = [data_c1; data_c2]; avg_coke = [avg_c1 avg_c2];
max_coke = [max_c1 max_c2]; min_coke = [min_c1 min_c2];

cd /cit/imaging/saliency/STIMtriangle/DBtraining;
result;
data_t1 = data; avg_t1 = avg_info; max_t1 = max_info; min_t1 = min_info;
cd ../DBtest;
result;
data_t2 = data; avg_t2 = avg_info; max_t2 = max_info; min_t2 = min_info;
data_tria = [data_t1; data_t2]; avg_tria = [avg_t1 avg_t2];
max_tria = [max_t1 max_t2]; min_tria = [min_t1 min_t2];

figure;
analyze_info(data_auto, avg_auto, min_auto, max_auto, 'w-');
hold on;
analyze_info(data_coke, avg_coke, min_coke, max_coke, 'w--');
hold on;
analyze_info(data_tria, avg_tria, min_tria, max_tria, 'w-.');
title('- autobahn, -- coke, -. triangle');