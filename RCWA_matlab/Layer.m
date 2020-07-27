classdef Layer
properties
    d, Kx, Kz, Kv
end
methods
    function obj = Layer(d)
        obj.d = d;
        obj.Kx = 0;
        obj.Kz = 0;
        obj.Kv = [0, 0, 0];
    end

    function res = eps(obj, p, q, n)
    end

    function res = epsxx1(obj, n)
    end

    function res = epsm(obj, p, q, mn)
        res = zeros(mn, mn);
        for i = 1:mn
            for j = 1:mn
                res(i, j) = obj.eps(p, q, i-j);
            end
        end
    end
    
    function res = getA(obj, m, k0, ki)
        % K = [Kx, 0, Kz]
        % k2m = [kix, kiy, 0] - m.*K = [kix-m.*Kx, kiy, -m.*Kz]
        % E = \sum (Sxm(z) + Sym(z) + Szm(z)).*exp(-1j.*k2m.*r)
        % H = sqrt(epsilon0/mu0) \sum (Uxm(z) + Uym(z) + Uzm(z)).*exp(-1j.*k2m.*r)
        % V = [Sx, Sy, Ux, Uy]
        % dV/dt = 1j .* k0 .* A .* V = W * diag(q) * inv(W) V
        kix = ki(1);
        kiy = ki(2);
        ms = -m:m;
        mn = 2.*m+1;
        Id = eye(mn);

        kxim = diag(kix-ms.*obj.Kx) ./ k0;
        k1ym = kiy .* Id ./ k0;
        kzim = diag(-ms.*obj.Kz) ./ k0;

        exxm = obj.epsm(1, 1, mn);
        exym = obj.epsm(1, 2, mn);
        exzm = obj.epsm(1, 3, mn);
        eyym = obj.epsm(2, 2, mn);
        eyzm = obj.epsm(2, 3, mn);
        ezzm = obj.epsm(3, 3, mn);
        ezzm_1 = round(inv(ezzm), 12);
        
        A11m = kzim + kxim * ezzm_1 * exzm;
        A12m =  kxim * ezzm_1 * eyzm;
        A13m = -kxim * ezzm_1 * k1ym;
        A14m = -Id + kxim * ezzm_1 * kxim;

        A21m = k1ym * ezzm_1 * exzm;
        A22m = kzim + k1ym * ezzm_1 * eyzm;
        A23m = Id - k1ym * ezzm_1 * k1ym;
        A24m = k1ym * ezzm_1 * kxim;

        A31m =  kxim * k1ym + exym - eyzm * ezzm_1 * exzm;
        A32m = -kxim * kxim + eyym - eyzm * ezzm_1 * eyzm;
        A33m = kzim + eyzm * ezzm_1 * k1ym;
        A34m = -eyzm * ezzm_1 * kxim;

        A41m =  k1ym * k1ym - exxm + exzm * ezzm_1 * exzm;
        A42m = -k1ym * kxim - exym + exzm * ezzm_1 * eyzm;
        A43m = -exzm * ezzm_1 * k1ym;
        A44m = kzim + exzm * ezzm_1 * kxim;

        A = round([
            A11m, A12m, A13m, A14m;
            A21m, A22m, A23m, A24m;
            A31m, A32m, A33m, A34m;
            A41m, A42m, A43m, A44m
        ], 10);
        res = A;
    end
    
    function [Qp, Qm, Wp, Vp, Wm, Vm] = getQW(obj, m, k0, kiv)
        A = obj.getA(m, k0, kiv);
        [W, q] = eig(A);
        Q = 1.j .* k0 .* round(diag(q), 12);
        pPredict = imag(Q) <= 0;
        mPredict = imag(Q) > 0;
        Qp = Q(pPredict);
        Qm = Q(mPredict);
        WVp = W(:, pPredict);
        WVm = W(:, mPredict);
        M = length(WVp);
        Wp = WVp(1:M/2, :);
        Vp = WVp(M/2+1:end, :);
        Wm = WVm(1:M/2, :);
        Vm = WVm(M/2+1:end, :);
    end
    
    function [res1, res2] = getTm(obj, m, k0, kiv)
        d = obj.d;
        ms = -m:m;
        res1 = struct();
        res2 = struct();
        [Qp, Qm, res1.Wp, res1.Vp, res2.Wm, res2.Vm] = obj.getQW(m, k0, kiv);
        K = @(z) diag(exp(-1j.*[-ms.*obj.Kz, -ms.*obj.Kz].*z));
        Xp = @(z) diag(exp(Qp.*z));
        Xm = @(z) diag(exp(Qm.*z));
        res2.A = K(d) * res1.Wp * Xp(d);
        res2.B = K(d) * res1.Vp * Xp(d);
        res1.E = K(-d) * res2.Wm * Xm(-d);
        res1.D = K(-d) * res2.Vm * Xm(-d);
    end
end
end
