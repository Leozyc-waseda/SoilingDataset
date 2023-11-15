function f = simpleTest(X)
fprintf('TRYING\n');
X
fprintf('.');
ff = -X(1) * X(2) * X(3);
fprintf('result %f\n',ff);
f = ff;