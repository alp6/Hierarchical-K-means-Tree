function [tree, tree_idx] = hi_kmeans(data, b, depth)
%%% hi_kmeans builts a hierarchical tree by using kmeans recursively
% data: SIFT features
% b: brach number
% depth: number of levels

[idx, C] = kmeans(data,b);
tree_idx(:,depth) = idx;
depth = depth - 1;
sub = [];
for i = 1:b
    if (size(data(idx ==i,:),1) >= b) && depth > 1
        [aux, aux_idx] = hi_kmeans(data(idx == i,:),b,depth);
        sub = [sub aux];
        tree_idx (idx ==i,1:size(aux_idx,2)) = aux_idx;
    elseif (size(data(idx ==i,:),1) >= b) && depth == 1
        [aux_idx, C] = kmeans(data(idx == i,:),b);
        aux = struct('centers', C, 'sub',[]);
        sub = [sub aux];
        tree_idx (idx ==i,1) = aux_idx;
    elseif (size(data(idx ==i,:),1) < b)
        aux = struct('centers', [],'sub', []);
        sub = [sub aux];  
        tree_idx (idx ==i,1:depth) = ones(size(data(idx ==i,:),1), depth);
    end    
    tree = struct('centers', C, 'sub', sub);
end
    
end

