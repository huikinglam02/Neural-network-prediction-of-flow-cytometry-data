% This sctipt aims to use flow cytometry data of human T cells treated in
% various manners to test whether machine learning ideas could be used for
% analysis and prediction.
% The data are 11 color fluorescence of human T cells. Inititally I used
% gaussian mixture model to fit the distribution, but unfortunately there
% isn't a significant subpopulation within the data.
% Now given the 11 color flow data, I ask whether I can predict whether the
% cells were treated with the use of neural network.

% Data loading
[filename,pathname] = uigetfile('*.xls*','Please put in the  data:','MultiSelect','on');
cd(pathname);
Si_f=size(filename);
data=[];

for k=1:Si_f(2)
    [num,txt,raw] = xlsread(char(filename(k)));

    Si=size(num);
    for i=Si(1):-1:1
        if ~isempty(find(num(i,:)==1))
            num(i,:)=[];
        end
    end
    num(:,Si(2)+1)=k;
    data=[data;num];
end

% Neural network training
%% Setup the parameters you will use for this exercise
input_layer_size  = Si(2);  % 20x20 Input Images of Digits
hidden_layer_size = Si(2);   % 25 hidden units
num_labels = Si_f(2);          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 1000);

%  You should also try different values of lambda
lambda = 0;

X=data(:,1:Si(2));
y=data(:,Si(2)+1);

% random permutation of indices
Si_X=size(X);
permorder=randperm(Si_X(1));
X_shuffled=X(permorder,:);
y_shuffled=y(permorder);

% Normalization
X_mean=mean(X,1);
X_std=std(X,1);
for i=1:Si_X
    X_shuffled(i,:)=(X_shuffled(i,:)-X_mean)./X_std;
end

% Divide the data set into 60 % training, 20 % CV and 20 % test
X_train=X_shuffled(1:floor(0.6*Si_X(1)),:);
y_train=y_shuffled(1:floor(0.6*Si_X(1)),:);
X_cv=X_shuffled(floor(0.6*Si_X(1))+1:floor(0.8*Si_X(1)),:);
y_cv=y_shuffled(floor(0.6*Si_X(1))+1:floor(0.8*Si_X(1)),:);
X_test=X_shuffled(floor(0.8*Si_X(1))+1:end,:);
y_test=y_shuffled(floor(0.8*Si_X(1))+1:end,:);

% Train the NN
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

% Cross validation
lambda_choice=[0 0.01 0.03 0.1 0.3 1 3 10 30 100];
Si_lambda=size(lambda_choice);

for i=1:Si_lambda(2)
    % Train the NN
    % Create "short hand" for the cost function to be minimized
    lambda=lambda_choice(i);
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X_cv, y_cv, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, nn_params, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    pred = predict(Theta1, Theta2, X_cv);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_cv)) * 100);
    accuracy(i)=mean(double(pred == y_cv)) * 100;
end

[val, ind]=max(accuracy);
lambda=lambda_choice(ind);
% Train 10000 times to achieve best accuracy
% Train the NN
% Create "short hand" for the cost function to be minimized
options = optimset('MaxIter', 10000);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_cv, y_cv, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, X_cv);
fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == y_cv)) * 100);

% Prediction accuracy accuracy on test set
pred = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

accuracy_condition=zeros(num_labels,1);
for i=1:length(accuracy_condition)
    accuracy_condition(i)=mean(double(pred(y_test==i) == y_test(y_test==i))) * 100;
end
bar(accuracy_condition);
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
set(gca,'FontWeight','bold','FontSize',18);
pbaspect([1 1 1]);
ylabel('accuracy (%)','FontWeight','bold','FontSize',20);
set(gca,'xticklabel',filename);
