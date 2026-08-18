% === 纯基本语法实现：批量把 .mat 文件里唯一的变量改名为文件名（无扩展名）===
% 当前文件夹下所有 .mat 文件
fileList = dir('*.mat');

for k = 1:numel(fileList)
    % 取出带扩展名的文件名和不带扩展名的基名
    fullName = fileList(k).name;                % 例如 data2025.mat
    dotPos   = find(fullName == '.', 1, 'last');% 最后一个 . 的位置
    if isempty(dotPos)
        continue;                               % 理论上不会出现
    end
    baseName = fullName(1:dotPos-1);            % 例如 data2025
    
    % 用 matfile 只读方式先探变量（不加载进内存）
    m = matfile(fullName, 'Writable', false);
    varList = who(m);                           % 仍然要用 who，这里无法完全避免
    clear m                                     % 立刻释放
    
    if isempty(varList)
        disp(['空文件，跳过: ' fullName]);
        continue;
    end
    
    % 取第一个变量（绝大多数情况只有一个）
    oldName = varList{1};
    if numel(varList) > 1
        disp(['多变量文件 ' fullName '，默认改第一个: ' oldName]);
    end
    
    % 如果名字已经正确，直接跳过
    if strcmp(oldName, baseName)
        disp(['已正确，跳过: ' fullName]);
        continue;
    end
    
    % === 关键：可写方式打开 → 复制 → 删除旧变量 → 立即重写文件 ===
    m = matfile(fullName, 'Writable', true);
    
    % 1. 新名字 = 旧变量（只复制引用，不占额外内存）
    m.(baseName) = m.(oldName);
    
    % 2. 彻底删除旧变量（先赋空再从文件中移除）
    m.(oldName) = [];          % 删除变量
    
    % 3. 为了保险，再用 save 强制重写一次（-v7.3 支持部分读写且会真正清除被删变量）
    temp = load(fullName, baseName);   % 只加载我们刚刚改好的那个变量
    save(fullName, '-struct', 'temp', '-v7.3');  % 重写文件，只保留新变量
    clear temp m
    
    disp(['完成: ' fullName '  中  ' oldName ' → ' baseName]);
end

disp('全部处理完毕！');