close all; clear; clc;
warning off

addpath(genpath('./methods/'));
addpath(genpath('./utils/'));

param.ds_dir = './datasets/';
param.rec_dir = './results';

ds_name={'FashionVC'};
nbits=[16 32 64];
test_times=1;
param.load_type='first_setting';    % param.load_type='second_setting';

%% parameters
param.alpha=1e-3;
param.beta=10000;
param.eta=1;
param.gamma=1;
param.xi=1e-4;
param.rho=1e-1;

param.max_iter=7;

count=1;
for t=1:test_times
    param.t=t;
    % DATASET
    for ds=1:length(ds_name)
        param.ds_name=ds_name{ds};
        [param,train,query]=load_dataset(param);
        % CODE LENGTH
        for nb=1:length(nbits)
            param.nbits=nbits(nb);
            fprintf('CODE LENGTH: %d\n',param.nbits);
            CHOICE(param,train,query);
            fprintf('COUNT: %d\n',count);
            count=count+1;
        end
    end
end

