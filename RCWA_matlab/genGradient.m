function lays = genGradient(layClass, n, d, px, pz0, pz1, chi, no, ne)
    % n: layer number
    % d: total depth
    % px: period x
    % pz0, pz1: period z
    d0 = d/n;
    d0s = (0:(n-1))/n*d;
    pzf = @(z) 1/(1/pz0 + (1/pz1-1/pz0)*(z/d));
    lays = cell(length(d0s));
    for i = 1:length(d0s)
        di = d0s(i);
        pz = pzf(di);
        lays{i} = layClass(d0, 2*pi/px, 2*pi/pz, chi, no, ne);
    end
end


