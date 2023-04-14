%% JS(x_i||x_j)*S_ij^(v)^2+-r*S_ij^(v)+lambda*||S||_*
%% s.t. $_j=1^N S_ij^(v)=1

clear
clc

load('BBC.mat');
gt=truelabel{1}';
truth=gt;
cls_num = length(unique(gt));

tic
%% Note: each column is an sample (same as in LRR)
%% 
%data preparation...
 %X{1} = X1; X{2} = X2; X{3} = X3;

 X=data;
 for v=1:4
    [X{v}]=NormalizeData(full(X{v}));
    X{v}(X{v}==0)=10e-6;
     %X{v} = zscore(X{v},1);
 end



K = length(X); N = size(X{1},2); %sample number

for k=1:K
    S{k} = zeros(N,N);
    G{k} = zeros(N,N);
    TI{k} = zeros(N,N);
end

ti = zeros(N*N*K,1);
g = zeros(N*N*K,1);
dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX = [N, N, K];
%set Default
parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;

% ---------- initilization for Z and F -------- %
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';      % Binary  HeatKernel
for k=1:K
    Z{k} = constructW(X{k}',options);
    Z{k} = full(Z{k});
    Z1{k} = Z{k}-diag(diag(Z{k}));         
    Z{k} = (Z1{k}+Z1{k}')/2;
    Z{k} = (Z{k}+Z{k}')/2;
    distX{k} = JS_distance(X{k});
end
G=Z;
S=Z;



Isconverg = 0;epson = 1e-7;
r=0.5;
lambda=5;
iter = 0;
mu = 0.001; max_mu = 10e10; pho_mu = 2;


while(Isconverg == 0)
    fprintf('----processing iter %d--------\n', iter+1);
    S_old=S;
    %% Update S
    for k=1:K
        S{k}     = (mu*G{k}-TI{k}+r)./(2*distX{k}+mu);
        S{k}     = S{k} - diag(diag(S{k}));
        for  ic = 1:N
            idx    = 1:N;
            idx(ic) = [];
            S{k}(ic,idx) = EProjSimplex_new(S{k}(ic,idx));
        end
    end
    
    %% Update G
 
    S_tensor = cat(3, S{:,:});
    TI_tensor = cat(3, TI{:,:});
    s = S_tensor(:);
    ti = TI_tensor(:);

    %twist-version
   [g, objV] = wshrinkObj(s + 1/mu*ti,lambda/mu,sX,0,3)   ;
    G_tensor = reshape(g, sX);
    G_old=G;
    for k=1:K
        G{k} = G_tensor(:,:,k);
    end
    
    %% update TI  mu

    for k=1:K
        TI{k} = TI{k}+mu*(S{k}-G{k});
    end
    mu = min(mu*pho_mu, max_mu);
    

    %record the iteration information
%     history.objval(iter+1)   =  objV;

    %% coverge condition
    Isconverg = 1;
     for k=1:K
         
         if (norm(S{k}-G{k},inf)>epson)
            history.norm_S_G = norm(S{k}-G{k},inf);
            fprintf(    'norm_S_G %7.10f    \n', history.norm_S_G);
            Isconverg = 0;
         end
        
         if (norm(G_old{k}-G{k},inf)>epson)
            history.norm_G = norm(G_old{k}-G{k},inf);
            fprintf(    'norm_G %7.10f    \n', history.norm_G);
            Isconverg = 0;
        end
               
        if (norm(S_old{k}-S{k},inf)>epson)
            history.norm_S = norm(S_old{k}-S{k},inf);
            fprintf(    'norm_S %7.10f    \n', history.norm_S);
            Isconverg = 0;
        end

    end

    if (iter>200)
        Isconverg  = 1;
    end
    iter = iter + 1;
end
toc
time_cost =toc;
SS = 0;
for k=1:K
    SS = SS + abs(S{k})+abs(S{k}');
end

for i=1:10
    C{i} = SpectralClustering(SS,cls_num);
    [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C{i});
    ACCi(i) = Accuracy(C{i},double(truth));
    [A nmii(i) avgenti(i)] = compute_nmi(truth,C{i});
    if (min(truth)==0)
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth+1,C{i});
    else
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C{i});
    end       
end
ACC(1)=mean(ACCi);ACC(2) = std(ACCi);
nmi(1) = mean(nmii); nmi(2) = std(nmii);
AR(1) = mean(ARi);AR(2) = std(ARi);
F(1) = mean(Fi); F(2) = std(Fi);
P(1) = mean(Pi); P(2) = std(Pi);
R(1) = mean(Ri); R(2) = std(Ri);
avgent(1) = mean(avgenti); avgent(2) = std(avgenti);
result.ACC(1)=mean(ACCi); result.ACC(2) = std(ACCi);
result.nmi(1) = mean(nmii); result.nmi(2) = std(nmii);
result.AR(1) = mean(ARi); result.AR(2) = std(ARi);
result.F(1) = mean(Fi); result.F(2) = std(Fi);
result.P(1) = mean(Pi); result.P(2) = std(Pi);
result.R(1) = mean(Ri); result.R(2) = std(Ri);
result


