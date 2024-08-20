function [param, train, query] = split_dataset(X, Y, L,param)
% X: original features
% Y: original text
% L: original labels
% param.nquery: the number of test points
num_class1=param.num_class1;
num_class2=param.num_class2;
[N, ~] = size(L);
load(fullfile(param.ds_dir,[param.ds_name '.mat']));
if strcmp(param.ds_name,'FashionVC')
    if strcmp(param.load_type, 'first_setting')
        %% split test and training set
        rng default
        param.seed=rng;
        R = randperm(N);
        nquery = param.nquery; iquery = R(1:nquery);
        ntrain = N - nquery; itrain = R(nquery+1:N);

        % randomize again
        rng default
        itrain = itrain(randperm(ntrain));

        rng default
        iquery = iquery(randperm(nquery));

        train.X=X(itrain,:);
        rng default
        train.anchorX=train.X(randsample(2000,2000),:);
        train.Y=Y(itrain,:);
        rng default
        train.anchorY=train.Y(randsample(2000,2000),:);
        train.L1=L(itrain,1:num_class1);
        train.L2=L(itrain,num_class1+1:end);
        train.size=ntrain;
        train.A1_2=A1_2;

        % select training data
        query.X=X(iquery,:);
        query.X=Kernelize(query.X,train.anchorX);
        query.Y=Y(iquery,:);
        query.Y=Kernelize(query.Y,train.anchorY);
        query.L1=L(iquery,1:num_class1);
        query.L2=L(iquery,num_class1+1:end);
        query.size=nquery;

    elseif strcmp(param.load_type, 'second_setting')
        L2=L(:,9:end);
        [~,L2_idx]=sort(sum(L2),'descend');
        L2=L2(:,L2_idx);
        L(:,9:end)=L2;

        A1_2_l=A1_2(1:8,9:end);
        A1_2_l=A1_2_l(:,L2_idx);
        A1_2(1:8,9:end)=A1_2_l;
        A1_2(9:end,1:8)=A1_2_l';

        A1_2_2=A1_2(9:end,9:end);

        A_tmp=A1_2_2;
        for i=1:27
            for j=1:27
                A1_2_2(L2_idx(i),L2_idx(j))=A_tmp(i,j);
            end
        end
        A1_2(9:end,9:end)=A1_2_2;

        train.nchunks=5;
        labels=linspace(9,35,27);
        seperate=cell(train.nchunks,1);
        seperate{1,1}=[26+num_class1,27+num_class1,22+num_class1,23+num_class1,7+num_class1,8+num_class1];
        seperate{2,1}=[24+num_class1,25+num_class1,5+num_class1,4+num_class1];
        seperate{3,1}=[18+num_class1,19+num_class1,20+num_class1,21+num_class1,2+num_class1,6+num_class1];
        seperate{4,1}=[15+num_class1,16+num_class1,17+num_class1,3+num_class1,9+num_class1,10+num_class1];
        seperate{5,1}=[11+num_class1,12+num_class1,1+num_class1,13+num_class1,14+num_class1];
        train.seperate=seperate;

        train.chunksize = cell(train.nchunks,1);
        train.test_chunksize = cell(train.nchunks,1);

        XTrain = cell(train.nchunks,1);
        YTrain = cell(train.nchunks,1);
        LTrain = cell(train.nchunks,1);

        XQuery = cell(train.nchunks,1);
        YQuery = cell(train.nchunks,1);
        LQuery = cell(train.nchunks,1);

        label_allow=[];
        last_found_idx=[];

        label_layer_and_allow=linspace(1,8,8);

        for l=1:train.nchunks
            label_allow=[label_allow seperate{l,1}];
            label_notallow=setdiff(labels,label_allow);
            idx_find_all=find(sum(L(:,label_notallow),2)==0);
            idx_find=setdiff(idx_find_all,last_found_idx);
            last_found_idx=idx_find_all;

            rng default
            R = randperm(size(idx_find,1));
            queryInds = R(1,1:floor(size(idx_find,1)*0.1));
            sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end);

            X_tmp=X(idx_find,:);
            Y_tmp=Y(idx_find,:);
            L_tmp=L(idx_find,:);
            L_all_tmp=L(idx_find,:);

            XTrain{l,1}=X_tmp(sampleInds,:);
            YTrain{l,1}=Y_tmp(sampleInds,:);
            if l==1
                rng default
                train.anchorX=XTrain{l,1}(randsample(2000,2000),:);
                rng default
                train.anchorY=YTrain{l,1}(randsample(2000,2000),:);
            end

            L1Train{l,1}=L_tmp(sampleInds,label_layer_and_allow);
            L2Train{l,1}=L_tmp(sampleInds,label_allow);

            XQuery{l,1}=X_tmp(queryInds,:);
            XQuery{l,1}=Kernelize(XQuery{l,1},train.anchorX);
            YQuery{l,1}=Y_tmp(queryInds,:);
            YQuery{l,1}=Kernelize(YQuery{l,1},train.anchorY);
            L1Query{l,1}=L_tmp(queryInds,label_layer_and_allow);
            L2Query{l,1}=L_tmp(queryInds,label_allow);

            train.chunksize{l,1}=size(sampleInds,2);
            train.test_chunksize{l,1}=size(queryInds,2);
        end

        train.X=XTrain;
        train.Y=YTrain;
        train.L1=L1Train;
        train.L2=L2Train;
        train.A1_2=A1_2;

        query.X=XQuery;
        query.Y=YQuery;
        query.L1=L1Query;
        query.L2=L2Query;
    end
elseif strcmp(param.ds_name,'Ssense')
    if strcmp(param.load_type, 'first_setting')
        %         %% split test and training set
        %         rng default
        %         param.seed=rng;
        %         R = randperm(N);
        %         nquery = param.nquery; iquery = R(1:nquery);
        %         ntrain = N - nquery; itrain = R(nquery+1:N);
        %
        %         % randomize again
        %         rng default
        %         itrain = itrain(randperm(ntrain));
        %         rng default
        %         iquery = iquery(randperm(nquery));
        %
        %         train.X=X(itrain,:);
        %         rng default
        %         train.anchorX=train.X(randsample(2000,2000),:);
        %         train.Y=Y(itrain,:);
        %         rng default
        %         train.anchorY=train.Y(randsample(2000,2000),:);
        %         train.L1=L(itrain,1:num_class1);
        %         train.L2=L(itrain,num_class1+1:end);
        %         train.size=ntrain;
        %         train.A1_2=A1_2;
        %
        %         % select training data
        %         query.X=X(iquery,:);
        %         query.X=Kernelize(query.X,train.anchorX);
        %         query.Y=Y(iquery,:);
        %         query.Y=Kernelize(query.Y,train.anchorY);
        %         query.L1=L(iquery,1:num_class1);
        %         query.L2=L(iquery,num_class1+1:end);
        %         query.size=nquery;
    elseif strcmp(param.load_type, 'second_setting')
        %         L2=L(:,5:end);
        %         [~,L2_idx]=sort(sum(L2),'descend');
        %         L2=L2(:,L2_idx);
        %         L(:,5:end)=L2;
        %
        %         A1_2_l=A1_2(1:4,5:end);
        %         A1_2_l=A1_2_l(:,L2_idx);
        %         A1_2(1:4,5:end)=A1_2_l;
        %         A1_2(5:end,1:4)=A1_2_l';
        %
        %         A1_2_2=A1_2(5:end,5:end);
        %
        %         A_tmp=A1_2_2;
        %         for i=1:28
        %             for j=1:28
        %                 A1_2_2(L2_idx(i),L2_idx(j))=A_tmp(i,j);
        %             end
        %         end
        %         A1_2(5:end,5:end)=A1_2_2;
        %
        %         train.nchunks=5;
        %         labels=linspace(5,32,28);
        %         seperate=cell(train.nchunks,1);
        %         seperate{1,1}=[4+num_class1,15+num_class1,16+num_class1,25+num_class1,20+num_class1,28+num_class1,27+num_class1];
        %         seperate{2,1}=[2+num_class1,14+num_class1,17+num_class1,24+num_class1,21+num_class1,26+num_class1];
        %         seperate{3,1}=[3+num_class1,10+num_class1,13+num_class1,18+num_class1,23+num_class1];
        %         seperate{4,1}=[6+num_class1,8+num_class1,12+num_class1,19+num_class1,22+num_class1];
        %         seperate{5,1}=[1+num_class1,5+num_class1,9+num_class1,11+num_class1,7+num_class1];
        %         train.seperate=seperate;
        %
        %         train.chunksize = cell(train.nchunks,1);
        %         train.test_chunksize = cell(train.nchunks,1);
        %
        %         XTrain = cell(train.nchunks,1);
        %         YTrain = cell(train.nchunks,1);
        %         LTrain = cell(train.nchunks,1);
        %
        %         XQuery = cell(train.nchunks,1);
        %         YQuery = cell(train.nchunks,1);
        %         LQuery = cell(train.nchunks,1);
        %
        %         label_allow=[];
        %         last_found_idx=[];
        %
        %         label_layer_and_allow=linspace(1,4,4);
        %
        %         for l=1:train.nchunks
        %             label_allow=[label_allow seperate{l,1}];
        %             label_notallow=setdiff(labels,label_allow);
        %             idx_find_all=find(sum(L(:,label_notallow),2)==0);
        %             idx_find=setdiff(idx_find_all,last_found_idx);
        %             last_found_idx=idx_find_all;
        %
        %             rng default
        %             R = randperm(size(idx_find,1));
        %             queryInds = R(1,1:floor(size(idx_find,1)*0.1));
        %             sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end);
        %
        %             X_tmp=X(idx_find,:);
        %             Y_tmp=Y(idx_find,:);
        %             L_tmp=L(idx_find,:);
        %             L_all_tmp=L(idx_find,:);
        %
        %             XTrain{l,1}=X_tmp(sampleInds,:);
        %             YTrain{l,1}=Y_tmp(sampleInds,:);
        %             if l==1
        %                 rng default
        %                 train.anchorX=XTrain{l,1}(randsample(2000,2000),:);
        %                 rng default
        %                 train.anchorY=YTrain{l,1}(randsample(2000,2000),:);
        %             end
        %
        %             L1Train{l,1}=L_tmp(sampleInds,label_layer_and_allow);
        %             L2Train{l,1}=L_tmp(sampleInds,label_allow);
        %
        %             XQuery{l,1}=X_tmp(queryInds,:);
        %             XQuery{l,1}=Kernelize(XQuery{l,1},train.anchorX);
        %             YQuery{l,1}=Y_tmp(queryInds,:);
        %             YQuery{l,1}=Kernelize(YQuery{l,1},train.anchorY);
        %             L1Query{l,1}=L_tmp(queryInds,label_layer_and_allow);
        %             L2Query{l,1}=L_tmp(queryInds,label_allow);
        %
        %             train.chunksize{l,1}=size(sampleInds,2);
        %             train.test_chunksize{l,1}=size(queryInds,2);
        %         end
        %
        %         train.X=XTrain;
        %         train.Y=YTrain;
        %         train.L1=L1Train;
        %         train.L2=L2Train;
        %         train.A1_2=A1_2;
        %         % select training data
        %         query.X=XQuery;
        %         query.Y=YQuery;
        %         query.L1=L1Query;
        %         query.L2=L2Query;
    end
else
    error('DATASET NAME: ERROR!\n');
end

end