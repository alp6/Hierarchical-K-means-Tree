% Analysis and Search of Visual Data
% Project 2
% Paula Días Banet & Andrea Lorenzo Polo


%% 2 FEATURE EXTRACTION %%%%

% % This section extract KP of the server and client images, which are
% already charged in the .mat files
% 
% %Get SIFT KP for server image and save them
% path = "Data2/server/";
% DB_imgs = dir(strcat(path, "*.jpg"));
% descriptors = [];
% features = [];
% num_features = [];
% for i = 1:size(DB_imgs,1)  
%     I = imread(strcat(path, DB_imgs(i).name));
%     [f,d] = vl_sift(single(rgb2gray(I)),'PeakThresh', 4.5 , 'edgethresh', 2.3) ;
%     
%     aux = split(DB_imgs(i).name,'_');    
%     obj_number = erase(aux{1}, 'obj');
%     
%     d(129,:) = str2num(obj_number);
%     d(130,:) = i;
%     descriptors = [descriptors d];
%     features = [features f];
%     num_features = [num_features size(f,2)];
%     
% end
% save('descriptors.mat', 'descriptors');
% avg = sum(num_features)/size(num_features,2);
% 
% %% Get SIFT KP for each server images and save them 
% path = "Data2/client/";
% Q_imgs = dir(strcat(path, "*.jpg"));
% descriptors_Q = [];
% features_Q = [];
% num_features_Q = [];
% for i = 1:size(Q_imgs,1)
%     
%     I = imread(strcat(path, Q_imgs(i).name));
%     [f_Q,d_Q] = vl_sift(single(rgb2gray(I)),'PeakThresh', 4.5 , 'edgethresh', 2.3) ;
%     
%     aux = split(Q_imgs(i).name,'_');    
%     obj_number = erase(aux{1}, 'obj');
%     
%     d_Q(129,:) = str2num(obj_number);
%     d_Q(130,:) = i;
%     descriptors_Q = [descriptors_Q d_Q];
%     features_Q = [features_Q f_Q];
%     num_features_Q = [num_features_Q size(f_Q,2)];
%     
% end
% save('descriptors_Q.mat', 'descriptors_Q');
% avg_Q = sum(num_features_Q)/size(num_features_Q,2);

%% 3. CONSTRUCTION VOC. TREE %%%%%%
clear all; clc; 
% Load server KP descriptors
S = load('descriptors_150k.mat');
d = double((S.descriptors)');
d = sortrows(d,129);
 
% Build voc. tree
b = 4;
depth = 5;
[tree, idx] = hi_kmeans(d(:,1:128),b, depth);
tree.depth = depth;
tree.b = b;
idx = flip(idx,2);
words = unique(idx,'rows');

% Compute TF-IDF weights for each visual word (w(i,j))
K = max(d(:,129));                      % # total number of obj
F = zeros(K,1);                         % # of words in object j 
Ki = zeros(1,size(words,1));            % # of obj containing word i
f = zeros(K,size(words,1));             % # of times word i appears in obj j

for i = 1:size(words,1)                                     %for each visual word i
    
    occurrences = sum( idx == words(i,:),2) == depth;    %in wich KP it occurs word i
    
    obj_occ = d(:, 129).*double(occurrences);            %which objs have that word
    
    for j = 1:K 
        f(j,i) = sum(obj_occ ==j);
        F(j,1) = F(j,1) + f(j,i); 
        if f(j,i)>0
            Ki(1,i) = Ki(1,i) + 1;  
        end
    end   
end

a = (f./F);
b = log2(K./Ki);
w = a*b';

%% 4 QUERYNG %%

% Get descriptors of the query images
C = load('descriptors_Q_150k.mat');
d_Q = double((C.descriptors_Q)');
% Select a percentage
perc = 0.9;
perm = randperm(size(d_Q,1)) ;
sel = perm(1:perc*size(d_Q,1));
d_Q = d_Q (sel,:);
% Order by objetc
d_Q = sortrows(d_Q, 129);

% Push every KP into the vocabulary tree and find which path follows 
clear query_paths 
for q = 1:size(d_Q,1)   
     aux = hi_push(tree, d_Q(q,1:128));
     query_paths(q, 1:size(aux,2)) = aux;
end


% Assign a visual word to each KP
kp_words = zeros(size(d_Q,1) ,1);
for k = 1:size(words,1) 
    index = sum (query_paths == words(k,:),2) == depth;
    kp_words = double(index).*k + kp_words;     
end

% Build score matrix
K_Q = max(d_Q(:,129));      % Number of query objects
score = zeros(K_Q, K);      % Scores of every query obj to every server obj

for k = 1:K_Q
    
    aux = zeros(1,K);
    objQ_words = double(d_Q(:,129) == k).*kp_words;
    
    for i = 1:size(words,1)
        rep_word(i) = sum(objQ_words == i); % how many times word i appears in query object k
        for j = 1:K
            a = (f(j,i)./F(j,1));
            b = log2(K./Ki(1,i));
            tfidf = a.*b;
            score(k,j)=  score(k,j) + tfidf*rep_word(i);
        end       
    end
end

% Compute Recall Rate
[max_score, max_idx] = max(score, [], 2);
corrects = sum(max_idx == ([1:50])');
recall_rate = corrects/K_Q

score_5 = score;
corrects_5 = 0;
for n = 1:5
    [max_score, max_idx] = max(score_5, [], 2);
    corrects_5 = corrects_5 + sum(max_idx == ([1:50])');
    score_5(:,max_idx) = 0;
end
recall_rate_5 = corrects_5/K_Q


