function [F, E, Ef] = Fp(xs, xss, K1, K2)

p1 = transpose([xs, ones(size(xs, 1), 1)]);
p2 = transpose([xss, ones(size(xss, 1), 1)]);
norm1 = getNormMat2d(p1);
norm2 = getNormMat2d(p2);


p1 = norm1 * p1;
p2 = norm2 * p2;

%p1 = p1(1:2,:);
%p2 = p2(1:2,:);
disp(size(xs));
%disp(p1(1:8,:));
disp(size(p1));

for n = 1: size(xs,2)
%     A(n, :) = kron(p1(n, :), p2(n, :));
    A(n, :) = [p2(n, 1) .* p1(n, 1),...
                p2(n, 1) .* p1(n, 2),...
                p2(n, 1),...
                p2(n, 2) .* p1(n, 1),...
                p2(n, 2) .* p1(n, 2),...
                p2(n, 2),...
                p1(n, 1),...
                p1(n, 2),...
                1];
end