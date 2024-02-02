clear;

%Load your spacetime data of interest
load("../network_output/tetradnet_test.mat");

[x,e] = find_linear_transformation(x,e);

%% statistics
figure(1);

tiledlayout(2,2);

nexttile
histogram(x);
title('x');

nexttile
%%{
N = size(e,1);
cond_e = zeros();
for i = 1:N
  cond_e(i) = cond(squeeze(e(i,:,:)));
end
semilogy(1:N, cond_e);
title('cond_e');


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


%% Make a good  graphic of curvature distributions
figure(2);
clf

histogram( riemann, 128, "Normalization", "countdensity", "EdgeAlpha", 0 );
hold on
histogram( ricci, 128, "Normalization", "countdensity", "EdgeAlpha", 0 );
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

tl = tiledlayout(4,4)

for i = 1:4
for j = 1:4
nexttile
ms = 30;
scatter3( x(:,2), x(:,3), x(:,4), 30, g(:,i,j), 'filled' );
pbaspect([1 1 1]);
%colorbar();
%clim([-1 1]);
xlabel("x^1", "rotation", 0);
ylabel("x^2", "rotation", 0);
zlabel("x^3", "rotation", 0);
xticks([-1 1]);
yticks(xticks);
zticks(xticks);
%title( ("g_{" + (i-1)) + (j-1) + "}" );
clim([-1 1]);
colorbar()
colormap bluewhitered
title(tl, "metric components $g_{\mu\nu}$", "interpreter", "latex", "fontsize", 30);
set(gcf, 'color', 'w');
drawnow
%saveas(gcf, ""+i+""+j+".png");
end
end

set( tl, "Padding", "compact" );
set( tl, "TileSpacing", "compact" );



function [x,e] = find_linear_transformation(x,e)

%Find a good coordinate transformation

N = size(e, 1);
g = zeros(N, 4, 4);
for i = 1:N
  e2 = squeeze( e(i,:,:) );
  g(i,:,:) = e2*diag([-1,1,1,1])*e2';
end

g0 = squeeze(mean(g));
[V,D] = eigs(g0);

[~,I] = sort(diag(D));

D2 = zeros(4,4);
V2 = zeros(4,4);
V = V*sqrt(abs(D));
for i = 1:4
  D2(i,i) = sign(D(I(i), I(i)));
  V2(:,i) = V(:,I(i));
end
D2;
V = V*sqrt(abs(D));

g0 - V*D*V';
g0 - V2*D2*V2';

%Transform x and e with a linear transformation
x = x*V2;
N = size(e,1);
for i = 1:N
  e(i,:,:) = V2\squeeze(e(i,:,:));
end
end