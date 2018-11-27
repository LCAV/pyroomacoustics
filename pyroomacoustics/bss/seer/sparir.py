def SpaRIR(G, S, delay, weights, q, gini):

    L = length(G)

    if rem(L, 2) == 1, error('The length of G must be even.') end

    S = S(1:L / 2 + 1) % S(k) and G(k)
    for k > L / 2 + 1 not used due to the symmetry of FFT (g is assumed to be real-valued)

    if ~exist('q', 'var'):
        q = 1

    if q > 1:
        G = [G(1:L / 2 + 1)zeros(q * L - L / 2 - 1, 1)]
        S = [S
        false(q * L - L / 2 - 1, 1)]
        L = q * L

    y = [real(G(S))
    imag(G(S))]
    M = length(y)

    if ~exist('delay', 'var'):
        delay = 20

    delay = q * delay

    if ~exist('gini', 'var'): # if no initialization is given
        g = zeros(L, 1)
        g(delay + 1) = 1
    else:
        g = gini

    if ~exist('weights', 'var'):
        tau = sqrt(L) / (y'*y) # * ones(L,1)
        tau = tau * exp(0.11 * abs(((1:L)
        '-delay)/q).^0.3)
    elif isempty(weights):
        tau = sqrt(L) / (y'*y)
        tau = tau * exp(0.11 * abs(((1:L)
        '-delay)/q).^0.3)
    elif length(weights) == 1:
        tau = ones(L, 1) * weights
    else:
        tau = reshape(repmat(weights
        ',q,1),L,1)

    maxiter = 10000
    alphamax = 1e5 % maximum
    step - length
    parameter
    alpha
    alphamin = 1e-7 % minimum
    step - length
    parameter
    alpha
    tol = 1e-4

    aux = zeros(L / 2, 1)
    G = fft(g)
    Ag = [real(G(S))
    imag(G(S))]
    r = Ag - y % instead
    of
    r = A * g - y
    aux(S(1: L / 2 + 1))=r(1: M / 2) + 1
    i * r(M / 2 + 1: end)
    gradq = L / 2 * ifft(aux, L, 'symmetric') % instead
    of
    gradq = A
    '*r
    alpha = 10
    support = g
    ~ = 0
    iter = 0

    crit = zeros(maxiter, 1)
    criterion = -tau(support). * sign(g(support)) - gradq(support)
    crit(iter + 1) = sum(criterion. ^ 2)

    while (crit(iter + 1) > tol) & & iter < maxiter:
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1 / alpha), tau / alpha)
        dg = g - prev_g
        DG = fft(dg)
        Adg = [real(DG(S))
        imag(DG(S))]
        r = prev_r + Adg % faster
        than
        A * g - y
        dd = dg(:)'*dg(:)
        dGd = Adg(:)'*Adg(:)
        alpha = min(alphamax, max(alphamin, dGd / (realmin + dd)))
    iter = iter + 1
    support = g
    ~ = 0
    aux(S(1: L / 2 + 1))=r(1: M / 2) + 1
    i * r(M / 2 + 1: end)
    gradq = L / 2 * ifft(aux, L, 'symmetric')
    criterion = -tau(support). * sign(g(support)) - gradq(support)
    crit(iter + 1) = sum(criterion. ^ 2) + sum(abs(gradq(~support)) - tau(~support) > tol)
    end

    if q > 1:
        g = q * resample(g, 1, q, 100)


    fprintf('SpaRIR: %d iterations done.\n', iter)

    if sum(g == 0) == L:
        disp('Zero solution! Do you really like it?')
        disp('Check input parameters such as weights, delay, gini or')
        disp('internal parameters such as tol, alphamax, alphamin, etc.')





def soft(x, T):
    if sum(abs(T(:))) == 0:
        y = x
    else:
        y = max(abs(x) - T, 0)
        # y = sign(x). * y
        y = y. / (y + T). * x
    return y