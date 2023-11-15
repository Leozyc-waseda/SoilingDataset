% Script to run the Simulink version of the saliency algorithm
file = '/lab/bruce/CT2WS/data/exImg002.jpg';
model = 'test3channel_c';
m_path = '/lab/bruce/saliency/src/Matlab/CT2WS';
s_path = '/lab/bruce/saliency/src/Matlab/Simulink/CT2WS';

addpath(m_path);
addpath(s_path);


% Load an example image
ib = imread(file);
figure;image(ib);title(file);axis equal
img = double(ib);

% Set up structures that the "from workspace" block likes...
red.time = 0;
red.signals.values = squeeze(img(:,:,1));
red.signals.dimensions = size(red.signals.values);
green.time = 0;
green.signals.values = squeeze(img(:,:,2));
green.signals.dimensions = size(green.signals.values);
blue.time = 0;
blue.signals.values = squeeze(img(:,:,3));
blue.signals.dimensions = size(blue.signals.values);

% Run the model
sim(model)

% Look at the result.
View(sal_img)