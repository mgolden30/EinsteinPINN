clear;

load("../solution.mat")

% Check the Wald p122 against the w one-forms we get from the network

p = 1; %pick a point

%Wald defines f and h
f = 1-2/x(p,2);
h = 1/f;

w02 = squeeze( w(p,:,1,3) )
w03 = squeeze( w(p,:,1,3) )

w12 = squeeze( w(p,:,2,3) )
w12a= [0,0,-1.0/sqrt(h),0]

w13 = squeeze(w(p,:,2,4))
w13a= [0,0,0,-1.0/sqrt(h)*sin(x(p,3))]

w12./w12a
w13./w13a

return

%% stats
tiledlayout(2,2);

nexttile
histogram(x);
title('x');

nexttile
%{
N = size(e,1);
cond_e = zeros();
for i = 1:N
  cond_e(i) = cond(squeeze(e(i,:,:)));
end
semilogy(1:N, cond_e);
title('cond_e');
%}
histogram(e)
title('e');

nexttile
histogram(ricci)
title('ricci');

nexttile
riemann( abs(riemann) < 1e-5) = nan; %get rid of pointless exact zeros
histogram(riemann)
title('riemann');

squeeze( w1(1,:,:,1)./w2(1,:,:,1) )
squeeze(ricci(1,:,:))

return;

%% Make a good  graphic of curvature distributions
clf

histogram( riemann, 128, "Normalization", "cdf", "EdgeAlpha", 0 );
hold on
histogram( ricci, 128, "Normalization", "cdf", "EdgeAlpha", 0 );
hold off

legend({"Riemann", "Ricci"});

%% visualize fields with color
clf

N = size(e, 1);
g = zeros(N, 4, 4);
for i = 1:N
  e2 = squeeze( e(i,:,:) );
  g(i,:,:) = e2*diag([-1,1,1,1])*e2';
end

for i = 1:4
for j = i:4
scatter3( x(:,2), x(:,3), x(:,4), 100, g(:,i,j), 'filled' );
pbaspect([1 1 1]);
colorbar();
%clim([-1 1]);
xlabel("x^1", "rotation", 0);
ylabel("x^2", "rotation", 0);
zlabel("x^3", "rotation", 0);
xticks([-1 1]);
yticks(xticks);
zticks(xticks);
title( ("g_{" + (i-1)) + (j-1) + "}" );
colormap jet
set(gcf, 'color', 'w');
drawnow
saveas(gcf, ""+i+""+j+".png");
end
end
%% Make a movie of 
N = size(e, 1);
g = zeros(N, 4, 4);
for i = 1:N
  e2 = squeeze( e(i,:,:) );
  g(i,:,:) = e2*diag([-1,1,1,1])*e2';
end

nt = 64;
sigma_t = 0.3;
ts = linspace(-1,1,nt);
for t = 1:nt
  clf;
  scatter3( [],[],[]);
  hold on
    
    ms = 100*exp( -(x(:,1) - ts(t)).^2 / sigma_t^2 )
    ms = ms + 0.1; %Can't have exactly zero
    scatter3( x(:,2), x(:,3), x(:,4), ms, g(:,2,2), 'filled' );
    
    colorbar();
    pbaspect([ 1 1 1]);
    drawnow
  hold off
end