function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% Part 1: 
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
a1 = [ones(1,m);X'];
z2 = Theta1*a1;
a2 = [ones(1,m);sigmoid(z2)];
z3 = Theta2*a2;
a3 = sigmoid(z3)';


% Cost function without regularization
J = trace(-y_matrix'*log(a3)-(1-y_matrix)'*log(1-a3))/m;

% Cost function with regularization
theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);
J = J + (sum(sum(theta1.^2))+sum(sum(theta2.^2)))*lambda/(2*m);

%Part 2: gradient without regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% m = the number of training examples (5000)
%%% n = the number of training features, including the initial bias
%%% unit. (401)
%%% h = the number of units in the hidden layer - NOT including the bias
%%% unit (25)
%%% r = the number of output classifications (10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3 = a3-y_matrix; % dimension(m x r)
d2 = d3 *theta2.*sigmoidGradient(z2'); % dimension (m x h)
Delta1 = (a1*d2)'; % dimension (h x n)
Delta2 = (a2*d3)'; %dimension (r x [h+1])
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

%Part 3: Gradient with Regularization:
%------------------------------------------
% 8: So, set the first column of Theta1 and Theta2 to all-zeros. 
% 9: Scale each Theta matrix by ?/m. Use enough parenthesis so the
% operation is correct. 
% 10: Add each of these modified-and-scaled Theta
% matrices to the un-regularized Theta gradients that you computed earlier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta1_new = Theta1;
Theta1_new(:,1) = 0;
Theta2_new = Theta2;
Theta2_new(:,1) = 0;
Theta1_new = lambda/m * Theta1_new;
Theta2_new = lambda/m * Theta2_new;
Theta1_grad = Theta1_grad + Theta1_new;
Theta2_grad = Theta2_grad + Theta2_new;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
