fp = fopen('result.txt','wt');



%% 
data = load('20w.txt');
 X = data(:, [1,3,5]); % X is a #OfExamScores x 2 matrix
 y = data(:, 10);      % y is a #OfExamScores x 1 matrix

%  
%  figure;
% for i=1:200000
%     plot3(X(i,1),X(i,2),X(i,3))
%     hold on;
% end    
 
     
 
%%
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];



% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, gradient] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', gradient);
 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%%



%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
% NOTE: that by using fminunc, you do not have to write any loops yourself,
% or set a learning rate like you did for gradient descent. You ONLY need
% to provide a function calculating the cost and the gradient.

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);



%%

%%%%%%%%%%%%%%%%%%predict the result of test data%%%%%%%%%%%%%%%%%%




datatest = load('test.txt')



Test =datatest(:, [1,3,5]);


[k, t] = size(Test);



Test = [ones(k, 1) Test];




prob = sigmoid(Test * theta)


fprintf(fp,'Id,Prediction\n');


for i = 1:2000
    fprintf(fp,'%d,%f\n',i,prob(i,:));
end  




% fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
%          'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);






