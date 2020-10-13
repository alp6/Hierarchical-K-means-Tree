function [path] = hi_push(tree,data)
%This function push a query feature into the given tree and outputs the
%path this feature follows until it reaches a leaf node.
%   Detailed explanation goes here

    
    path = [];
    path_c = 'tree.centers';
    path_str = 'tree.sub';
    
    for i = 1:tree.depth
        center_aux = eval(path_c);
        d = [];
        d = vecnorm(data - double(center_aux),2,2);
        min_idx = find(d == min(d));
        path(i) = min_idx(1);
        path_c;
        path_str = [ path_str(1:size(path_str,2)-3), 'sub(', num2str(path(i)), ').sub'];
        str_aux = eval(path_str);
        path_c = [ path_c(1:size(path_c,2)-7), 'sub(', num2str(path(i)), ').centers'];
        
        if isempty(str_aux)
            if isempty(eval(path_c))
                path(i+1) = 1;
            else             
                center_aux = eval(path_c);
                d = [];
                d = vecnorm(data - double(center_aux),2,2);
                min_idx = find(d == min(d));
                path(i+1) = min_idx(1);
            end
            break
    end

end

