function train = train_CHOICE(param,train,idx,first_round,round)
tic;
if strcmp(param.load_type, 'first_setting')
    r = param.nbits;
    num_class1=param.num_class1;
    num_class2=param.num_class2;

    dx=size(train.anchorX,1);
    dy=size(train.anchorY,1);

    n_t=numel(idx);

    %% hyperparameters
    alpha=param.alpha;
    beta=param.beta;
    eta=param.eta;
    gamma=param.gamma;
    xi=param.xi;
    rho=param.rho;

    max_iter = param.max_iter;

    X_new=train.X(idx,:);          % image feature vector
    X_new = Kernelize(X_new,train.anchorX)';
    Y_new=train.Y(idx,:);          % text feature vector
    Y_new = Kernelize(Y_new,train.anchorY)';
    L1_new=train.L1(idx,:);         % labels at the 1st layer
    L2_new=train.L2(idx,:);         % labels at the 2nd layer
    dc_old=size(L1_new,2)+size(L2_new,2);
    A1_2=train.A1_2;

    if first_round==true
        train.B=[];
        train.V=[];
        train.accX=[];
        train.accY=[];
        train.GTrain=[];
        train.trained=0;
        train.time.train_time=[];

        %% initialize tempory variables
        T=cell(1,11);

        %% dx
        % M1 = B_new * X_new' + [B_old * X_old']
        T{1,1} = zeros(r,dx);

        % M2 = X_new * X_new' + [X_old * X_old']
        T{1,2} = zeros(dx,dx);

        %% dy
        % M5 = B_new * Y_new' + [B_old * Y_old']
        T{1,5} = zeros(r,dy);

        % M6 = Y_new * Y_new' + [Y_old * Y_old']
        T{1,6} = zeros(dy,dy);

        % M3 = B_new * GTrain_new' + [B_old * G_old']
        T{1,3} = zeros(r,dc_old);

        % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
        T{1,4} = zeros(dc_old,dc_old);

        % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
        T{1,7} = zeros(dc_old,dx);

        % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
        T{1,8} = zeros(dc_old,dy);

        % M9 = R'*R
        T{1,9}=zeros(r,r);

        % M10 = V_new * V_new' + [V_old * V_old']
        T{1,10}=zeros(r,r);

        % M11 = V_new * GTrain_new' + [V_old * GTrain_old']
        T{1,11}=zeros(r,dc_old);

        B_old=zeros(r,param.chunk_size);
        V_old=zeros(r,param.chunk_size);
        X_old=zeros(dx,n_t);
        Y_old=zeros(dy,n_t);
        GTrain_old=zeros(dc_old,param.chunk_size);
    else
        T=train.T;
        B_old=train.B;
        V_old=train.V;
        X_old=train.accX;
        Y_old=train.accY;
        GTrain_old=train.GTrain;
    end

    rng default;
    B_new=sign(randn(r,n_t));
    rng default;
    V_new=randn(r,n_t);
    rng default;
    Z_b=sign(randn(r,n_t));
    Q = B_new - Z_b;

    P=zeros(r,dc_old);
    Wx=zeros(r,dx);
    Wy=zeros(r,dy);

    %% Hybrid Semantic Matrix
    L_new=[L1_new,L2_new];

    GTrain_new = (A1_2*L_new')';
    GTrain_new = (GTrain_new ./ sum(GTrain_new.^2,2).^0.5)';

    % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
    T{1,4} = T{1,4} + GTrain_new * GTrain_new';

    % M2 = X_new * X_new' + [X_old * X_old']
    T{1,2} = T{1,2} + X_new * X_new';

    % M6 = Y_new * Y_new' + [Y_old * Y_old']
    T{1,6} = T{1,6} + Y_new * Y_new';

    % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
    T{1,7} = T{1,7} + GTrain_new * X_new';

    % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
    T{1,8} = T{1,8} + GTrain_new * Y_new';

    % online hash code learning
    for i = 1:max_iter
        %%  V-step
        % M3 = B_new * GTrain_new' + [B_old * GTrain_old']
        T{1,3} = B_old * GTrain_old' + B_new * GTrain_new';

        Z=B_new+alpha*r*T{1,3}*GTrain_new;

        Temp = Z*Z'-(1/n_t)*Z*ones(n_t,1)*ones(1,n_t)*Z';
        [~,Lmd,GG] = svd(Temp);clear Temp

        idx = (diag(Lmd)>1e-6);
        O = GG(:,idx);
        O_ = orth(GG(:,~idx));
        N = Z'*O/(sqrt(Lmd(idx,idx)))-(1/n_t)*ones(n_t,1)*(ones(1,n_t)*Z')*O/(sqrt(Lmd(idx,idx)));
        rng default;
        N_ = orth(randn(n_t,r-length(find(idx==1))));
        V_new = sqrt(n_t)*[O O_]*[N N_]';

        %% P-step
        % M3 = B_new * GTrain_new' + [B_old * GTrain_old']
        % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
        % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
        % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
        P=(1/(beta+eta*gamma))*((beta*T{1,3} + eta*gamma*Wx*T{1,7}'+ eta*gamma*Wy*T{1,8}')/T{1,4});

        %% Wx and Wy-step
        % M2 = X_new * X_new' + [X_old * X_old']
        % M1 = B_new * X_new' + [B_old * X_old']
        T{1,1}= B_new * X_new' + B_old * X_old';

        % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
        % online hash function learning
        Wx = (T{1,1}+gamma*P*T{1,7})/((1+gamma)*T{1,2}+xi*eye(dx));

        % M6 = Y_new * Y_new' + [Y_old * Y_old']
        % M5 = B_new * Y_new' + [B_old * Y_old']
        T{1,5}= B_new * Y_new' + B_old * Y_old';

        % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
        % online hash function learning
        Wy = (T{1,5}+gamma*P*T{1,8})/((1+gamma)*T{1,6}+xi*eye(dy));

        %% B_new-step
        % M10 = V_new * V_new' + [V_old * V_old']
        T{1,10}=V_new * V_new' + V_old * V_old';

        % M11 = V_new * GTrain_new' + [V_old * GTrain_old']
        T{1,11}=V_new * GTrain_new' + V_old * GTrain_old';

        B_new=sign(2*V_new-alpha*T{1,10}*Z_b+2*alpha*r*T{1,11}*GTrain_new+2*beta*P*GTrain_new+2*eta*Wx*X_new+2*eta*Wy*Y_new+rho*Z_b-Q);

        %% Z_b-step
        % M10 = V_new * V_new' + [V_old * V_old']
        Z_b=sign(rho*B_new+Q-alpha*T{1,10}*B_new);

        %% Q-step
        Q=Q+rho*(B_new-Z_b);

    end

    train.T=T;
    train.Wx=Wx;
    train.Wy=Wy;
    if first_round==true
        train.B = B_new;
        train.V = V_new;
        train.accX = X_new;
        train.accY = Y_new;
        train.GTrain = GTrain_new;
    else
        train.B = [train.B B_new];
        train.V = [train.V V_new];
        train.accX = [train.accX X_new];
        train.accY = [train.accY Y_new];
        train.GTrain = [train.GTrain GTrain_new];
    end
    train.trained=train.trained+n_t;
    train.time.train_time=[train.time.train_time;toc];

elseif strcmp(param.load_type, 'second_setting')
        r = param.nbits;
        num_class1=param.num_class1;
        num_class2=param.num_class2;

        dx=size(train.anchorX,1);
        dy=size(train.anchorY,1);

        n_t=size(train.Y{round,1},1);

        %% hyperparameters
        alpha=param.alpha;
        beta=param.beta;
        eta=param.eta;
        gamma=param.gamma;
        xi=param.xi;
        rho=param.rho;

        max_iter = param.max_iter;

        X_new=train.X{round,1};          % image feature vector
        X_new = Kernelize(X_new,train.anchorX)';
        Y_new=train.Y{round,1};          % text feature vector
        Y_new = Kernelize(Y_new,train.anchorY)';
        L1_new=train.L1{round,1};         % labels at the 1st layer
        L2_new=train.L2{round,1};         % labels at the 2nd layer

        if first_round==true
            A1_2=train.A1_2(train.Label_allow,train.Label_allow);               
            dc_old=length(train.Label_allow);
            train.B=[];
            train.V=[];
            train.accX=[];
            train.accY=[];
            train.GTrain=[];
            train.trained=0;
            train.time.train_time=[];

            %% initialize tempory variables
            T=cell(1,11);

            %% dx
            % M1 = B_new * X_new' + [B_old * X_old']
            T{1,1} = zeros(r,dx);

            % M2 = X_new * X_new' + [X_old * X_old']
            T{1,2} = zeros(dx,dx);

            %% dy
            % M5 = B_new * Y_new' + [B_old * Y_old']
            T{1,5} = zeros(r,dy);

            % M6 = Y_new * Y_new' + [Y_old * Y_old']
            T{1,6} = zeros(dy,dy);

            % M3 = B_new * GTrain_new' + [B_old * G_old']
            T{1,3} = zeros(r,dc_old);

            % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
            T{1,4} = zeros(dc_old,dc_old);

            % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
            T{1,7} = zeros(dc_old,dx);

            % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
            T{1,8} = zeros(dc_old,dy);

            % M9 = R'*R
            T{1,9}=zeros(r,r);

            % M10 = V_new * V_new' + [V_old * V_old']
            T{1,10}=zeros(r,r);

            % M11 = V_new * GTrain_new' + [V_old * GTrain_old']
            T{1,11}=zeros(r,dc_old);

            B_old=zeros(r,n_t);
            V_old=zeros(r,n_t);
            X_old=zeros(dx,n_t);
            Y_old=zeros(dy,n_t);
            GTrain_old=zeros(dc_old,n_t);

        else
            A1_2=train.A1_2(setdiff(train.Label_allow, train.seperate{round,1}),train.Label_allow);                
            dc_old=length(setdiff(train.Label_allow, train.seperate{round,1}));
            T=train.T;
            B_old=train.B;
            V_old=train.V;
            X_old=train.accX;
            Y_old=train.accY;
            GTrain_old=train.GTrain;

            % M4 = T{1,4} = zeros(dc_old,dc_old);
            T{1,4} = [T{1,4} zeros(size(T{1,4},1),dc_old-size(T{1,4},2));zeros(dc_old-size(T{1,4},1),dc_old)];    

            % M3 = T{1,3} = zeros(r,dc_old)
            T{1,3} = [T{1,3},zeros(r,dc_old-size(T{1,3},2))];

            % M7 = T{1,7} = zeros(dc_old,dx);
            T{1,7} = [T{1,7};zeros(dc_old-size(T{1,7},1),dx)];

            % M8 = T{1,8} = zeros(dc_old,dy);
            T{1,8} = [T{1,8};zeros(dc_old-size(T{1,8},1),dy)];

            % M11 = T{1,11}=zeros(r,dc_old);
            T{1,11} = [T{1,11},zeros(r,dc_old-size(T{1,11},2))];
        end

        rng default;
        B_new=sign(randn(r,n_t));
        rng default;
        V_new=randn(r,n_t);
        rng default;
        Z_b=sign(randn(r,n_t));
        Q = B_new - Z_b;

        P=zeros(r,dc_old);
        Wx=zeros(r,dx);
        Wy=zeros(r,dy);

        %% Hybrid Semantic Matrix
        L_new=[L1_new,L2_new];

        GTrain_new = (A1_2*L_new')';    
        GTrain_new = (GTrain_new ./ sum(GTrain_new.^2,2).^0.5)';

        % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
        T{1,4} = T{1,4} + GTrain_new * GTrain_new';

        % M2 = X_new * X_new' + [X_old * X_old']
        T{1,2} = T{1,2} + X_new * X_new';

        % M6 = Y_new * Y_new' + [Y_old * Y_old']
        T{1,6} = T{1,6} + Y_new * Y_new';

        % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
        T{1,7} = T{1,7} + GTrain_new * X_new';

        % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
        T{1,8} = T{1,8} + GTrain_new * Y_new';

        % online hash code learning
        for i = 1:max_iter
            %%  V-step
            % M3 = B_new * GTrain_new' + [B_old * GTrain_old']
            tmp_BG = B_old * GTrain_old';
            temp_T1_3_old = [ tmp_BG,zeros(r,dc_old-size(tmp_BG,2))];
            T{1,3} = temp_T1_3_old + B_new * GTrain_new';

            Z=B_new+alpha*r*T{1,3}*GTrain_new;

            Temp = Z*Z'-(1/n_t)*Z*ones(n_t,1)*ones(1,n_t)*Z';
            [~,Lmd,GG] = svd(Temp);clear Temp

            idx = (diag(Lmd)>1e-6);
            O = GG(:,idx);
            O_ = orth(GG(:,~idx));

            N = Z'*O/(sqrt(Lmd(idx,idx)))-(1/n_t)*ones(n_t,1)*(ones(1,n_t)*Z')*O/(sqrt(Lmd(idx,idx)));
            rng default;
            N_ = orth(randn(n_t,r-length(find(idx==1))));

            V_new = sqrt(n_t)*[O O_]*[N N_]';

            %% P-step
            % M3 = B_new * GTrain_new' + [B_old * GTrain_old']
            % M4 = GTrain_new * GTrain_new' + [GTrain_old * GTrain_old']
            % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
            % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
            P=(1/(beta+eta*gamma))*((beta*T{1,3} + eta*gamma*Wx*T{1,7}'+ eta*gamma*Wy*T{1,8}')/T{1,4});

            %% Wx and Wy-step
            % M2 = X_new * X_new' + [X_old * X_old']
            % M1 = B_new * X_new' + [B_old * X_old']
            T{1,1}= B_new * X_new' + B_old * X_old';
            % M7 = GTrain_old * X_old' + [GTrain_new * X_new']
            % online hash function learning
            Wx = (T{1,1}+gamma*P*T{1,7})/((1+gamma)*T{1,2}+xi*eye(dx));

            % M6 = Y_new * Y_new' + [Y_old * Y_old']
            % M5 = B_new * Y_new' + [B_old * Y_old']
            T{1,5}= B_new * Y_new' + B_old * Y_old';
            % M8 = GTrain_old * Y_old' + [GTrain_new * Y_new']
            % online hash function learning
            Wy = (T{1,5}+gamma*P*T{1,8})/((1+gamma)*T{1,6}+xi*eye(dy));

            %% B_new-step
            % M10 = V_new * V_new' + [V_old * V_old']
            T{1,10}=V_new * V_new' + V_old * V_old';

            % M11 = V_new * GTrain_new' + [V_old * GTrain_old']
            tmp_VG=V_old * GTrain_old';
            temp_T1_11_old = [ tmp_VG,zeros(r,dc_old-size(tmp_VG,2))];
            T{1,11}= temp_T1_11_old + V_new * GTrain_new';

            B_new=sign(2*V_new-alpha*T{1,10}*Z_b+2*alpha*r*T{1,11}*GTrain_new+2*beta*P*GTrain_new+2*eta*Wx*X_new+2*eta*Wy*Y_new+rho*Z_b-Q);

            %% Z_b-step
            % M10 = V_new * V_new' + [V_old * V_old']
            Z_b=sign(rho*B_new+Q-alpha*T{1,10}*B_new);

            %% Q-step
            Q=Q+rho*(B_new-Z_b);

        end

        train.T=T;
        train.Wx=Wx;
        train.Wy=Wy;
        if first_round==false
            % GTrain_old=zeros(dc_old,n_t);
            GTrain_old = [GTrain_old;zeros(dc_old-size(GTrain_old,1),size(GTrain_old,2))];
            train.GTrain = [GTrain_old GTrain_new];
            train.B = [train.B B_new];
            train.V = [train.V V_new];
            train.accX = [train.accX X_new];
            train.accY = [train.accY Y_new];
        else
            train.GTrain = GTrain_new;
            train.B = B_new;
            train.V = V_new;
            train.accX = X_new;
            train.accY = Y_new;
        end
        train.trained=train.trained+n_t;
        train.time.train_time=[train.time.train_time;toc];
end
end