classdef RCWA
properties
    n1, n3, base1, base2, m, ms, M, layers,
    phi, theta, k1uv, DERl, DERr, DETl, DETr
end
methods
    % epsilon = \sum epsilon \exp(1j.*m.*K.*r)
    function obj = RCWA(n1, n3, layers, m, base1, base2)
        obj.n1 = n1; % refractive index of input region
        obj.n3 = n3; % refractive index of output region
        obj.base1 = norml(base1).'; % electric field for mode 1
        obj.base2 = norml(base2).'; % electric field for mode 2
        % e.g., TM: [1, 0], TE: [0, 1], LC: [1, 1j], RC: [1j, 1]
        obj.m = m; % RCWA order
        obj.ms = -m:m;
        obj.M = length(obj.ms);
        obj.layers = layers; % lay
    end
    
    function [q, W] = getLayerQW(obj, k0, kiv)
        % A = W * diag(q) * inv(W)

        Am = obj.layers.getA(obj.m, k0, kiv);
        [W, q] = eig(Am);
        q = 1.j .* k0 .* round(diag(q), 12);
    end

    function [DERl, DERr, DETl, DETr] = solve(obj, phi, theta, wl, Eu, Ev)
        % phi: projected angle between x axis in x-y plane 
        % theta: angle between z axis
        % wl: wavelength unit: m
        % [Eu, Ev]: input electric field, 
        % e.g., TM: [1, 0], TE: [0, 1], LC: [1, 1j], RC: [1j, 1]
        % light to z+
        obj.phi = deg2rad(phi);
        obj.theta = deg2rad(theta);
        obj.k1uv = [
            eulerMatrix({obj.phi, obj.theta, 0}, 1, 3),...
            eulerMatrix({obj.phi, obj.theta, 0}, 2, 3),...
            eulerMatrix({obj.phi, obj.theta, 0}, 3, 3)...
        ];
        II = [0, -1; 1, 0];

        obj.DERl = zeros(1, obj.M);
        obj.DERr = zeros(1, obj.M);
        obj.DETl = zeros(1, obj.M);
        obj.DETr = zeros(1, obj.M);
        [n1, n3] = deal(obj.n1, obj.n3);
        Ei = norml([Eu, Ev]).';
        k0k = 2.*pi./wl;
        k0v = obj.k1uv .* k0k;
        kiv = k0v .* n1;
        k1k = k0k .* n1;
        k3k = k0k .* n3;
        [Q, W] = obj.getLayerQW(k0k, kiv);

        k1iv = zeros(length(obj.ms), 3);
        for i = 1:length(k1iv)
            k1iv(i, :) = kiv - obj.ms(i)*obj.layers.Kv;
            kx = k1iv(i, 1);
            ky = k1iv(i, 2);
            if kx.^2 + ky.^2 <= k1k.^2
                k1iv(i, end) = -sqrt(k1k.^2 - kx.^2 - ky.^2);
            else
                k1iv(i, end) = 1j.*sqrt(-k1k.^2 + kx.^2 + ky.^2);
            end
        end
        UVk1ixym = zeros(2*obj.M, 2);
        for i = 1:obj.M
            k1v = k1iv(i, :);
            UVk1ixym(2*i-1:2*i, :) = k2uv_xy(k1v);
        end
        UVk1ixm = UVk1ixym(1:2:end, :);
        UVk1iym = UVk1ixym(2:2:end, :);
        SRlxm = diag(UVk1ixm * obj.base1);
        SRlym = diag(UVk1iym * obj.base1);
        SRrxm = diag(UVk1ixm * obj.base2);
        SRrym = diag(UVk1iym * obj.base2);
        URlxm = n1 .* diag(UVk1ixm * II * obj.base1);
        URlym = n1 .* diag(UVk1iym * II * obj.base1);
        URrxm = n1 .* diag(UVk1ixm * II * obj.base2);
        URrym = n1 .* diag(UVk1iym * II * obj.base2);

        k3iv = zeros(length(obj.ms), 3);
        for i = 1:length(k3iv)
            k3iv(i, :) = kiv - obj.ms(i)*obj.layers.Kv;
            kx = k3iv(i, 1);
            ky = k3iv(i, 2);
            if kx.^2 + ky.^2 <= k3k.^2
                k3iv(i, end) = sqrt(k3k.^2 - kx.^2 - ky.^2);
            else
                k3iv(i, end) = -1j.*sqrt(-k3k.^2 + kx.^2 + ky.^2);
            end
        end
        UVk3ixym = zeros(2*obj.M, 2);
        for i = 1:obj.M
            k3v = k3iv(i, :);
            UVk3ixym(2*i-1:2*i, :) = k2uv_xy(k3v);
        end
        UVk3ixm = UVk3ixym(1:2:end, :);
        UVk3iym = UVk3ixym(2:2:end, :);
        STlxm = diag(UVk3ixm * obj.base1);
        STlym = diag(UVk3iym * obj.base1);
        STrxm = diag(UVk3ixm * obj.base2);
        STrym = diag(UVk3iym * obj.base2);
        UTlxm = n3 .* diag(UVk3ixm * II * obj.base1);
        UTlym = n3 .* diag(UVk3iym * II * obj.base1);
        UTrxm = n3 .* diag(UVk3ixm * II * obj.base2);
        UTrym = n3 .* diag(UVk3iym * II * obj.base2);

        d = obj.layers.d;
        kziv = -obj.ms.*obj.layers.Kv(end);

        ZERO = zeros(obj.M, obj.M);
        pPredict = imag(Q) <= 0;
        mPredict = imag(Q) > 0;
        Qp = Q(pPredict);
        Qm = Q(mPredict);
        Wp = W(1:end, pPredict);
        Wm = W(1:end, mPredict);
        eQpzm = @(z) diag(exp(Qp.*z));
        eQmzm = @(z) diag(exp(Qm.*(z-d)));
        ekiz = @(z) diag(exp(-1j.*kziv.*z));
        seg = length(Wp)/4;
        Wsxm0 = Wp(1:seg, :);
        Wsym0 = Wp(seg+1:2*seg, :);
        Wuxm0 = Wp(2*seg+1:3*seg, :);
        Wuym0 = Wp(3*seg+1:end, :);
        Wsxm1 = Wm(1:seg, :);
        Wsym1 = Wm(seg+1:2*seg, :);
        Wuxm1 = Wm(2*seg+1:3*seg, :);
        Wuym1 = Wm(3*seg+1:end, :);
        Wsxm = {Wsxm0, Wsxm1};
        Wsym = {Wsym0, Wsym1};
        Wuxm = {Wuxm0, Wuxm1};
        Wuym = {Wuym0, Wuym1};
        Wm = @(Wm, z) ekiz(z) * [Wm{1} * eQpzm(z), ekiz(-d) * Wm{2} * eQmzm(z)];

        P = [...
            SRlxm, SRrxm, ZERO,  ZERO,  -Wm(Wsxm, 0);
            SRlym, SRrym, ZERO,  ZERO,  -Wm(Wsym, 0);
            ZERO,  ZERO,  STlxm, STrxm, -Wm(Wsxm, d);
            ZERO,  ZERO,  STlym, STrym, -Wm(Wsym, d);
            URlxm, URrxm, ZERO,  ZERO,  -Wm(Wuxm, 0);
            URlym, URrym, ZERO,  ZERO,  -Wm(Wuym, 0);
            ZERO,  ZERO,  UTlxm, UTrxm, -Wm(Wuxm, d);
            ZERO,  ZERO,  UTlym, UTrym, -Wm(Wuym, d)...
        ];

        kuvxy = [...
            eulerMatrix({obj.phi, obj.theta, 0}, 1, 1),...
            eulerMatrix({obj.phi, obj.theta, 0}, 1, 2); 
            eulerMatrix({obj.phi, obj.theta, 0}, 2, 1),...
            eulerMatrix({obj.phi, obj.theta, 0}, 2, 2)...
        ];
        delta = (obj.ms == 0).';
        UVkiixym = zeros(2*obj.M, 2);
        for i = 1:2:obj.M
            UVkiixym(i:i+1, :) = kuvxy;
        end
        UVkiixm = UVkiixym(1:2:end, :);
        UVkiiym = UVkiixym(2:2:end, :);
        ZERO = zeros(obj.M, 1);
        p = [
            -1 .* delta .* (UVkiixm * Ei);
            -1 .* delta .* (UVkiiym * Ei);
            ZERO;
            ZERO;
            -1 .* delta .* n1 .* (UVkiixm * II * Ei);
            -1 .* delta .* n1 .* (UVkiiym * II * Ei);
            ZERO;
            ZERO
        ];
        res = P\p;
        seg = length(res)/8;
        rlv = res(1:seg);
        rrv = res(seg+1:2*seg);
        tlv = res(2*seg+1:3*seg);
        trv = res(3*seg+1:4*seg);
        obj.DERl = -abs(rlv).^2 .* real(k1iv(1:end, end))./kiv(end);
        obj.DERr = -abs(rrv).^2 .* real(k1iv(1:end, end))./kiv(end);
        obj.DETl =  abs(tlv).^2 .* real(k3iv(1:end, end))./kiv(end);
        obj.DETr =  abs(trv).^2 .* real(k3iv(1:end, end))./kiv(end);
        DERl = obj.DERl;
        DERr = obj.DERr;
        DETl = obj.DETl;
        DETr = obj.DETr;
    end
end
end
    