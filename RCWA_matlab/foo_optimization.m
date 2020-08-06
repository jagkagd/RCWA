function loss = foo_optimization(ps)
    d1 = ps(1);
    Kz1 = ps(2);
    d2 = ps(3);
    Kz2 = ps(4);
    %Kx = ps(5);
    ng = 1.56;
    dn = 0.2;
    no = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. - dn./2;
    ne = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. + dn./2;
    alpha = rad2deg(asin(1/ng*sin(deg2rad(20))))./2;
    wl0 = 940e-9;
    k0 = 2.*pi./wl0;
    Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha));
    Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha));
    thetas = -40:2:40;
    nn = 1;
    DETr = [];
    lays = {PVG2(wl0*d1/dn, Kx, Kz1, -1, no, ne), PVG2(wl0*d2/dn, Kx, Kz2, -1, no, ne)};
    for i = 1:length(thetas)
        theta = rad2deg(asin(sin(deg2rad(thetas(i)))/ng));

        rcwa = RCWA(ng, ng, lays, nn, [1, 1j], [1, -1j]);
        [~, ~, ~, detr, ~] = rcwa.solve(0, theta, wl0, 1, 1j);
        DETr(i, :) = detr;
    end

    DETr = DETr.';
    nn = nn + 1;
    loss = sum(1-DETr(nn-1, :));

end

