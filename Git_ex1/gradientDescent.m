function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    x = X(:,2);
    
    prediction = theta(1) + theta(2) *x;
    error = prediction - y;
    
    theta_zero = theta(1) - alpha * (1/m) * sum(error);
    theta_one  = theta(2) - alpha * (1/m) * sum((error).*x);
     theta = ([theta_zero, theta_one]');
    
%======================= wrong one I wrote==================================
%prediction = X*theta; %sample code from vectorized implementation
%Errors = (prediction - y);

%inside_sum = Errors' * X(:,2);

%inside_sum_2 = (1/m) * sum(inside_sum);
 
%theta = (theta - alpha * inside_sum_2);
 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
  disp(min(J_history));
end
