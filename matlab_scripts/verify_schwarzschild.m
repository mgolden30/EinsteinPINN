%{
Load the n
%}

clear;

load("../network_output/schwarzschild_tetradnet_train.mat")

fprintf("\n\nThis script will check if our curvature tensors are being computed as expected...\n");

N = size(e,1);
g = 0*e;
for i = 1:N
  g( i, :, : ) = squeeze( e(i,:,:) ) * diag([-1,1,1,1]) * squeeze( e(i,:,:) );
end

%Take the first data-point and explicitly compare calculations
p = 1; %test particle index

r = x(p,2);
th= x(p,3);

f = 1 - 2/r; 
df= -2/r/r;

threshold = 1e-4; %these are done in single precision
check_if_equal = @(x,y) assert( abs(x-y) < threshold );

%Metric components
check_if_equal( g(p,1,2), 0 );
check_if_equal( g(p,1,3), 0 );
check_if_equal( g(p,1,4), 0 );
check_if_equal( g(p,2,3), 0 );
check_if_equal( g(p,2,4), 0 );
check_if_equal( g(p,3,4), 0 );
check_if_equal( g(p,1,1), -(1-2/r) );
check_if_equal( g(p,2,2), 1/(1-2/r) );
check_if_equal( g(p,3,3), r^2 );
check_if_equal( g(p,4,4), r^2 * sin(th)^2 );
fprintf("metric is correct!\n");



%Check some components of the connection one-form
check_if_equal( w(p,3,2,3), -sqrt(f) ); % equation 6.1.17 (alpha3 = 0)
fprintf("connection one-form is correct!\n");



%Check the Ricci tensor vanishes
for i = 1:4
  for j = 1:4
    check_if_equal( ricci(p,i,j), 0 ); %Ricci should vanish for Scharzschild
  end
end
fprintf("Ricci tensor is correct!\n");



%Check a non-zero element of the Riemann tensor. Importantly, all indices
%Have been projected into Minkowski space. So the expressions in Wald need
%to be weighted by the inverse tetrad. 
check_if_equal( riemann(p,3,4,3,4), (1-f)/r/r );  % R_{th ph th ph} 6.1.34
check_if_equal( riemann(p,1,3,1,3), -df/r/2);     % R_{t  th t  th} 6.1.30
check_if_equal( riemann(p,1,2,1,2), -4/r/r/r/2 ); % R_{t  r  t  r } 6.1.29
fprintf("Riemann tensor is correct!\n\n");

fprintf("Passed all tests!\n");