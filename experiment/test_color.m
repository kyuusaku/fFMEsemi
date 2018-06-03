% test color 

h = figure(1);
m_size = 15;
line_width = 2;

line_style = {'-','--',':','-.'};
line_marker = {'o','+','*','.','x','s','d','^','v','>','<','p','h'};
line_color = {'y','m','c','r','g','b','w','k'};

x = 1:10;
n = numel(x);
str_legend = {};
count = 0;
% for i = 1:numel(line_style)
%     for j = 1:numel(line_marker)
%         for k = 1:numel(line_color)
%             count = count + 1
%             str_type = strcat(line_style{i}, line_marker{j}, line_color{k});
%             str_legend{count} = str_type;
%             plot(x, ones(1,n)*count, str_type, ...
%                 'LineWidth', line_width, 'MarkerSize', m_size);
%             hold on;
%         end
%     end
% end
for i = 1:1
    for j = 1:1
        for k = 1:numel(line_color)
            count = count + 1
            str_type = strcat(line_style{i}, line_marker{j}, line_color{k});
            str_legend{count} = str_type;
            plot(x, ones(1,n)*count, str_type, ...
                'LineWidth', line_width, 'MarkerSize', m_size);
            hold on;
        end
    end
end
set(gca,'YLim',[0,9]);
set(gca,'XLim',[0,11]);
legend(str_legend, ...
    'Location','EastOutside');
print(h,'-dpng','test_color.png');

%%
im = imread('test_color.png');
gray = rgb2gray(im);
imshow(gray);