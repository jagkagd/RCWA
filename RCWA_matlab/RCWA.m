classdef RCWA
properties
    n1, n3, base1, base2, m, ms, M, layers,
    phi, theta, k0uv, DERl, DERr, DETl, DETr,
    k1iv, k3iv,
    lKx
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
        Kxs = [];
        for i = 1:length(obj.layers)
            Kxs(i) = obj.layers{i}.Kx;
        end
        if ~all(Kxs == Kxs(1))
            throw(MException("RCWA:layers", "All layers should have the same Kx"))
        end
        obj.lKx = Kxs(1);
    end
    
    function res = getLayerT(obj, n1, k0, kiv)
        for i = 1:length(obj.layers)
            layer = obj.layers{i};
            [res1, res2] = layer.getTm(obj.m, k0, kiv);
            res{i} = {res1, res2};
        end
    end

    function [res1, res2] = getIn(obj, k1iv, Ei)
        II = [0, -1; 1, 0];
        ZERO = zeros(obj.M);
        n1 = obj.n1;

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
        
        res1 = struct();
        res2 = struct();

        SI = [
            [diag(delta .* (UVkiixm * Ei)), ZERO],
            [ZERO, diag(delta .* (UVkiiym * Ei))]
        ];
        SR = [
            [SRlxm, SRrxm],
            [SRlym, SRrym]
        ];
        UI = n1 * [
            [diag(delta .* (UVkiixm * II * Ei)), ZERO],
            [ZERO, diag(delta .* (UVkiiym * II * Ei))]
        ];
        UR = [
            [URlxm, URrxm],
            [URlym, URrym]
        ];
        res2.A = SI;
        res2.Wm = SR;
        res2.B = UI;
        res2.Vm = UR;
    end
    
    function [res1, res2] = getOut(obj, k3iv)
        II = [0, -1; 1, 0];
        n3 = obj.n3;

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
        ST = [
            [STlxm, STrxm],
            [STlym, STrym]
        ];
        UT = [
            [UTlxm, UTrxm],
            [UTlym, UTrym]
        ];
        ZERO2 = zeros(2*obj.M);
        
        res1 = struct();
        res2 = struct();
        
        res1.Wp = ST;
        res1.Vp = UT;
        res1.E = ZERO2;
        res1.D = ZERO2;
    end

    function [DERl, DERr, DETl, DETr, obj] = solve(obj, phi, theta, wl, Eu, Ev)
        % phi: projected angle between x axis in x-y plane 
        % theta: angle between z axis
        % wl: wavelength unit: m
        % [Eu, Ev]: input electric field, 
        % e.g., TM: [1, 0], TE: [0, 1], LC: [1, 1j], RC: [1j, 1]
        % light to z+
        obj.phi = deg2rad(phi);
        obj.theta = deg2rad(theta);
        obj.k0uv = [
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
        k0v = obj.k0uv .* k0k;
        kiv = k0v .* n1;
        k1k = k0k .* n1;
        k3k = k0k .* n3;

        k1iv = zeros(length(obj.ms), 3);
        for i = 1:length(k1iv)
            k1iv(i, :) = kiv - obj.ms(i)*[obj.lKx, 0, 0];
            kx = k1iv(i, 1);
            ky = k1iv(i, 2);
            if kx.^2 + ky.^2 <= k1k.^2
                k1iv(i, end) = -sqrt(k1k.^2 - kx.^2 - ky.^2);
            else
                k1iv(i, end) = 1j.*sqrt(-k1k.^2 + kx.^2 + ky.^2);
            end
        end

        k3iv = zeros(length(obj.ms), 3);
        for i = 1:length(k3iv)
            k3iv(i, :) = kiv - obj.ms(i)*[obj.lKx, 0, 0];
            kx = k3iv(i, 1);
            ky = k3iv(i, 2);
            if kx.^2 + ky.^2 <= k3k.^2
                k3iv(i, end) = sqrt(k3k.^2 - kx.^2 - ky.^2);
            else
                k3iv(i, end) = -1j.*sqrt(-k3k.^2 + kx.^2 + ky.^2);
            end
        end
        Tl0 = obj.getLayerT(obj.n1, k0k, kiv);
        Tl = [Tl0{:}];
        [~, resIn] = obj.getIn(k1iv, Ei);
        [resOut, ~] = obj.getOut(k3iv);
        Ts = {resIn, Tl{:}, resOut};

        ZERO = zeros(2*obj.M);

        res = cell(length(Ts), 1);
        for i = 1:2:length(Ts)
            T1 = Ts{i};
            T2 = Ts{i+1};
            ZEROS = [ZERO, ZERO];
            left = repmat(ZEROS, 1, (i-1)/2);
            right = repmat(ZEROS, 1, length(Ts)/2-1-(i-1)/2);
            res{i, 1}   = [left, T1.A, T1.Wm, -T2.Wp, -T2.E, right];
            res{i+1, 1} = [left, T1.B, T1.Vm, -T2.Vp, -T2.D, right];
        end
        P0 = cell2mat(res);
        P = P0(:, (2*obj.M+1):(end-2*obj.M));

        SI = resIn.A;
        UI = resIn.B;
        p = [-diag(SI).', -diag(UI).', repmat(diag(ZERO).', 1, 2*length(obj.layers))].';
        
        res = P\p;
        rlv = res(1:obj.M);
        rrv = res(obj.M+1:2*obj.M);
        tlv = res(end-2*obj.M+1:end-obj.M);
        trv = res(end-obj.M+1:end);
        
        obj.DERl = -abs(rlv).^2 .* real(k1iv(1:end, end))./kiv(end);
        obj.DERr = -abs(rrv).^2 .* real(k1iv(1:end, end))./kiv(end);
        obj.DETl =  abs(tlv).^2 .* real(k3iv(1:end, end))./kiv(end);
        obj.DETr =  abs(trv).^2 .* real(k3iv(1:end, end))./kiv(end);
        obj.k1iv = k1iv;
        obj.k3iv = k3iv;
        DERl = obj.DERl;
        DERr = obj.DERr;
        DETl = obj.DETl;
        DETr = obj.DETr;
    end
end
end
    