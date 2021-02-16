ng = 1.56;
dn = 0.2;
no = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. - dn./2;
ne = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. + dn./2;
wl0 = 940e-9;
k0 = 2.*pi./wl0;
alpha = rad2deg(asin(1/ng*sin(deg2rad(60))))./2;
Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha));
Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha));

% let d1 = wl*dd1/dn, d2 = wl*dd2/dn
% four initial parameters: dd1, Kz1, dd2, Kz2
x0 = [0.481681478, 0.5, 0.392741029, 0.5];
options = optimoptions('fmincon','Algorithm','interior-point','Display','iter');
% results in res
res = fmincon(@foo_optimization, x0, [], [], [], [], [0, -10, 0, -10], [1, 10, 1, 10], [], options);

thetas = -40:40;
nn = 1;
DERl = [];
DERr = [];
DETl = [];
DETr = [];
DETr1 = [];
DETr2 = [];
lay1 = PVG2(wl0*res(1)/dn, Kx, res(2)*Kz, -1, no, ne);
lay2 = PVG2(wl0*res(3)/dn, Kx, res(4)*Kz, -1, no, ne);
lays = {lay1, lay2};
for i = 1:length(thetas)
    theta = rad2deg(asin(sin(deg2rad(thetas(i)))/ng));
    
    rcwa =  RCWA(ng, ng, lays, nn, [1, 1j], [1, -1j]);
    rcwa1 = RCWA(ng, ng, {lay1}, nn, [1, 1j], [1, -1j]);
    rcwa2 = RCWA(ng, ng, {lay2}, nn, [1, 1j], [1, -1j]);
    [derl, derr, detl, detr, rcwa] = rcwa.solve(0, theta, wl0, 1, 1j);
    [~, ~, ~, detr1, rcwa1] = rcwa1.solve(0, theta, wl0, 1, 1j);
    [~, ~, ~, detr2, rcwa2] = rcwa2.solve(0, theta, wl0, 1, 1j);
    DERl(i, :) = derl;
    DERr(i, :) = derr;
    DETl(i, :) = detl;
    DETr(i, :) = detr;
    DETr1(i, :) = detr1;
    DETr2(i, :) = detr2;

end

DERl = DERl.';
DERr = DERr.';
DETl = DETl.';
DETr = DETr.';
DETr1 = DETr1.';
DETr2 = DETr2.';

xxs = thetas;
nn = nn + 1;
ax1 = subplot(2, 1, 1);
figure(1);
hold(ax1, 'on');
plot(ax1, xxs, DERl(nn, :), 'r');
plot(ax1, xxs, DERl(nn-1, :), 'g');
plot(ax1, xxs, DERl(nn+1, :), 'b');
plot(ax1, xxs, DERr(nn, :), 'r--');
plot(ax1, xxs, DERr(nn-1, :), 'g--');
plot(ax1, xxs, DERr(nn+1, :), 'b--');
plot(ax1, xxs, DERl(nn+1, :)./(DERl(nn+1, :)+DETl(nn, :)), 'k');
hold(ax1, 'off');
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');
plot(ax2, xxs,  DETl(nn, :), 'r');
plot(ax2, xxs,  DETl(nn-1, :), 'g');
plot(ax2, xxs,  DETl(nn+1, :), 'b');
plot(ax2, xxs,  DETr(nn, :), 'r--');
plot(ax2, xxs,  DETr(nn-1, :), 'g--');
plot(ax2, xxs,  DETr(nn+1, :), 'b--');

plot(ax2, xxs,  DETr1(nn-1, :), 'k');
plot(ax2, xxs,  DETr2(nn-1, :), 'k--');
hold(ax2, 'off');