% stack the SLP images
fd = 'logs/SLP_demos';
mods = ['depth', 'IR', 'PM'] ;   % mod list
N= 1620;     % can also comes from list

for i = 1:3     % create the sub path
    fdd(i) = fullfile(fd, mods(i));

% make out folder
out_fd = 'log/stk_allC';     % all cover stack
if ~exist(out_fd, 'dir')
       mkdir(out_fd)
end
% list all files
n_end = 1 % test purpose

for i =1:n_end
    % format the pth
    % read int the images
    %  set the slice
    % set the dpi  normalize 0 ,1
    % show it
    % save it


