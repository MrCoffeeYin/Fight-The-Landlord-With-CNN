clear ; close all; clc


%% ��ʼ��ȫ�ֱ���
input_layer_size  = 548;
hidden_layer_size = 30;
num_labels = 1;


%% ��������
data = csvread("../data/data2.csv");
y = data(:, 1);
X = data(:, 2:end-1);
m = size(X, 1);
fprintf("�����������\n")


%% ��ʼ������
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
Theta3 = randInitializeWeights(hidden_layer_size, num_labels);
nn_params = [Theta1(:) ; Theta2(:); Theta3(:)];
fprintf("��ʼ���������\n")


%% ѵ��ģ��
options = optimset('MaxIter', 500);
lambda = 0;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):(hidden_layer_size * (input_layer_size + hidden_layer_size + 2))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
Theta3 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + hidden_layer_size + 2))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf("ѵ��ģ�����\n")


%% Ԥ��
pred = predict(Theta1, Theta2, Theta3, X);
fprintf('Ԥ��׼ȷ��: %f\n', mean(double(pred == y)) * 100);


%% �洢���
f = fopen("../data/theta1_2.csv", "w");
for i = 1:hidden_layer_size
    for j = 1:input_layer_size + 1
        fprintf(f, "%f,", Theta1(i, j));
    end
    fprintf(f, "\n");
end
f = fopen("../data/theta2_2.csv", "w");
for i = 1:hidden_layer_size
    for j = 1:hidden_layer_size + 1
        fprintf(f, "%f,", Theta2(i, j));
    end
    fprintf(f, "\n");
end
f = fopen("../data/theta3_2.csv", "w");
for i = 1:hidden_layer_size + 1
    fprintf(f, "%f\n", Theta3(1, i));
end
fprintf("�洢������\n");