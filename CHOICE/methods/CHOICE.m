% close all; clear; clc;
function CHOICE(param,train,query)

close all

if strcmp(param.load_type, 'first_setting')
    nchunks= floor(train.size/param.chunk_size);
elseif strcmp(param.load_type, 'second_setting')
    nchunks=5;
end


%% train
eva_info=cell(nchunks,1);

first_round=true;

if strcmp(param.load_type, 'first_setting')
    for i=1:nchunks
        idx_strt=(i-1)*param.chunk_size+1;

        if(i~=nchunks)
            idx_end=idx_strt-1+param.chunk_size;
        else
            idx_end=train.size;
        end

        train = train_CHOICE(param,train,idx_strt:idx_end,first_round,i);

        first_round = false;

        fprintf('-------------- Round / Total: %d / %d --------------\n',i,nchunks);
        eva_info{i,1}=evaluate_perf(train.B',query.X*train.Wx',query.Y*train.Wy',train.L2(1:train.trained,:),query.L2);
        fprintf('MAP in I->T: %.4g\n',eva_info{i,1}.map_image2text);
        fprintf('MAP in T->I: %.4g\n',eva_info{i,1}.map_text2image);
    end
elseif strcmp(param.load_type, 'second_setting')
    for i=1:nchunks
        if i==1
            train.Label_allow=[1:param.num_class1,train.seperate{i,1}];
            accL2=[];
            accWRSL2=[];
            idx_strt=1;
            idx_end=size(train.X{i,1},1)-1;
        else
            train.Label_allow=[train.Label_allow,train.seperate{i,1}];
            idx_strt=train.idx_end+1;
            idx_end=idx_strt+size(train.X{i,1},1)-1;
        end
        train.idx_strt=idx_strt;
        train.idx_end=idx_end;

        train = train_CHOICE(param,train,idx_strt:idx_end,first_round,i);

        first_round = false;

        fprintf('-------------- Round / Total: %d / %d --------------\n',i,nchunks);
        accL2=[accL2 zeros(size(accL2, 1), size(train.L2{i,1},2) - size(accL2, 2));train.L2{i,1}];
        eva_info{i,1}=evaluate_perf(train.B',query.X{i,1}*train.Wx',query.Y{i,1}*train.Wy',accL2,query.L2{i,1});
        fprintf('MAP in I->T: %.4g\n',eva_info{i,1}.map_image2text);
        fprintf('MAP in T->I: %.4g\n',eva_info{i,1}.map_text2image);
    end
end

fprintf('----------------------- Done -----------------------\n');

time=train.time;

end
