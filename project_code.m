clear
format compact
close all

load Salinas_Data

[p,n,l]=size(Salinas_Image); % Size of the Salinas cube

% Making a two dimensional array whose rows correspond to the pixels and
% the columns to the bands, containing only the pixels with nonzero label.
X_total=reshape(Salinas_Image, p*n,l);
L=reshape(Salinas_Labels,p*n,1);
existed_L=(L>0);   %This contains 1 in the positions corresponding to pixels with known class label
X=X_total(existed_L,:);
[px,nx]=size(X); % px= no. of rows (pixels) and nx=no. of columns (bands)


%calculating the pca for X 
[coeff,score,latent] = pca(X);
% Extract first two principal components
pca_data = score(:,1:2);



%% Hierarchichal clustering


%% WARD
Z = linkage(X,'ward','euclidean');
cl_label1 = cluster(Z,'maxclust',8);
%cr_tab=crosstab(cl_label1,y);
figure(1),
dendrogram(Z);
title('Dendogram-WARD')


cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label1;
im_cl_label1=reshape(cl_label_tot,p,n);
figure(2),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label1), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('Ward')

%ploting the clusters with respect to the first two principal componets 
figure(3),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label1),title('WARD');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');



%Calculating the normalized mutual Information and the Rand index
nmi1 = nmi(L(existed_L),cl_label1);
ri1 = rand_index(L(existed_L), cl_label1, 'adjusted');


%% Complete-link

Z = linkage(X,'complete','euclidean');
cl_label2 = cluster(Z,'maxclust',8);
%cr_tab=crosstab(cl_label2,y);
figure(4),
dendrogram(Z);
title('Dendogram-Complete link')


cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label2;
im_cl_label2=reshape(cl_label_tot,p,n);
figure(5),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label2), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('Complete link')

%ploting the clusters with respect to the first two principal componets 
figure(6),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label2),title('Complete link');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');

%Calculating the normalized mutual Information and the Rand index
nmi2 = nmi(L(existed_L),cl_label2);
ri2 = rand_index(L(existed_L), cl_label2, 'adjusted');



%% Weighted average distance (WPGMA)
 Z = linkage(X,'weighted','euclidean');
 cl_label3 = cluster(Z,'maxclust',8);
 figure(7),
 %cr_tab=crosstab(cl_label3,y);
 dendrogram(Z);
 title('WPGMA')

cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label3;
im_cl_label3=reshape(cl_label_tot,p,n);
figure(8),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label3), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('WPGMA')

%ploting the clusters with respect to the first two principal componets 
figure(9)
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label3),title('WPGMA');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');

%Calculating the normalized mutual Information and the Rand index
nmi3 = nmi(L(existed_L),cl_label3);
ri3 = rand_index(L(existed_L), cl_label3, 'adjusted');

%% CFO's

%% Probabilistic
k=8;
options = statset('MaxIter',1000);
gm = fitgmdist(X,k,'Options',options); % k is the number of clusters
cl_label4 = cluster(gm,X);
cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label4;
im_cl_label4=reshape(cl_label_tot,p,n);

figure(10),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label4), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('Probabilistic')

%ploting the clusters with respect to the first two principal componets 
figure(11),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label4),title('Probabilistic');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');

%Calculating the normalized mutual Information and the Rand index
nmi4 = nmi(L(existed_L),cl_label4);
ri4 = rand_index(L(existed_L), cl_label4, 'adjusted');





%% K-means
k=8;
cl_label5=kmeans(X,k);
cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label5;
im_cl_label5=reshape(cl_label_tot,p,n);
figure(12),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label5), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('K-means')

%ploting the clusters with respect to the first two principal componets 
figure(13),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label5),title('k-means');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');


%Calculating the normalized mutual Information and the Rand index
nmi5 = nmi(L(existed_L),cl_label5);
ri5 = rand_index(L(existed_L), cl_label5, 'adjusted');



%% Fuzzy c-means
k=8;
[ U, C ] = fuzzy_c_means(X, k, 2, 1e-9 );
[~,cl_label6]=max(U,[],2);
cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label6;
im_cl_label6=reshape(cl_label_tot,p,n);
figure(14),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label6), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle(' fuzzy c-means')

%ploting the clusters with respect to the first two principal componets 
figure(15),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label6),title('Fuzzy c-means');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');

%Calculating the normalized mutual Information and the Rand index
nmi6 = nmi(L(existed_L),cl_label6);
ri6 = rand_index(L(existed_L), cl_label6, 'adjusted');




%ploting the clusters with respect to the first two principal componets 
figure(16),
colormap("jet")
gscatter(pca_data(:,1),pca_data(:,2),L(existed_L));
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA -Labels')


%%  Possibilistic c-means


%Initializing
m = 8;           % Number of clusters
eta = rand(1,m); % Eta parameters of the clusters
q = 1;           % q parameter of the algorithm
sed = 1;         % Seed for random generator
init_proc = 2;   % Use "rand_data_init" initialization procedure
e_thres = 0.001; % Threshold for termination condition
X=transpose(X);

[U,theta]=possibi(X,m,eta,q,sed,init_proc,e_thres);

[m,cl_label7]=max(U,[],2);
cl_label_tot=zeros(p*n,1);
cl_label_tot(existed_L)=cl_label7;
im_cl_label7=reshape(cl_label_tot,p,n);
figure(17),
colormap("jet")
subplot(1,2,1), imagesc(im_cl_label6), axis equal
subplot(1,2,2), imagesc(Salinas_Labels), axis equal
sgtitle('Possibilistic c-means ')

%ploting the clusters with respect to the first two principal componets 
figure(18),
colormap("jet")
gscatter(pca_data(:,1),pca_data(:,2),cl_label7);
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('PCA -Possibilistic c-means ')

%Calculating the normalized mutual Information and the Rand index
nmi7 = nmi(L(existed_L),cl_label7);
ri7 = rand_index(L(existed_L), cl_label7, 'adjusted');

figure(19),
colormap("jet")
subplot(1,2,1),gscatter(pca_data(:,1),pca_data(:,2),cl_label7),title('Possibilistic c-means');
xlabel('First Principal Component');
ylabel('Second Principal Component');
subplot(1,2,2),gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), title('Original');
xlabel('First Principal Component');
ylabel('Second Principal Component');
% plotting all the results 



nmi=[nmi1,nmi2,nmi3,nmi4,nmi5,nmi6,nmi7];
ri=[ri1,ri2,ri3,ri4,ri5,ri6,ri7];

figure(20)
subplot(4,2,1), imagesc(Salinas_Labels), title('Original Image'), axis equal
subplot(4,2,2), imagesc(im_cl_label1), title('Ward'), axis equal
subplot(4,2,3), imagesc(im_cl_label2), title('Complete -Link'), axis equal
subplot(4,2,4), imagesc(im_cl_label3),title('WPGMA'), axis equal
subplot(4,2,5), imagesc(im_cl_label4),title('Probabilistic'), axis equal
subplot(4,2,6), imagesc(im_cl_label5),title('K-means'), axis equal
subplot(4,2,7), imagesc(im_cl_label6),title('Fuzzy c-means'), axis equal
subplot(4,2,8), imagesc(im_cl_label7),title('Possibilistic c-means'), axis equal


figure(21)
subplot(4,2,1), gscatter(pca_data(:,1),pca_data(:,2),L(existed_L)), axis equal
subplot(4,2,2), gscatter(pca_data(:,1),pca_data(:,2),cl_label1), axis equal
subplot(4,2,3), gscatter(pca_data(:,1),pca_data(:,2),cl_label2), axis equal
subplot(4,2,4), gscatter(pca_data(:,1),pca_data(:,2),cl_label3), axis equal
subplot(4,2,5), gscatter(pca_data(:,1),pca_data(:,2),cl_label4), axis equal
subplot(4,2,6), gscatter(pca_data(:,1),pca_data(:,2),cl_label5), axis equal
subplot(4,2,7), gscatter(pca_data(:,1),pca_data(:,2),cl_label6), axis equal
subplot(4,2,8), gscatter(pca_data(:,1),pca_data(:,2),cl_label7), axis equal





