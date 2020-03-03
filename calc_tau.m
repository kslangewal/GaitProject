% This function calculates tau, the time constant of the gating variables 
% of the Hodgkin Huxley equations. 
%
% Input:
% alpha:    The function handle to the alpha equation.
% beta:     The function handle to the beta equation.
% V:        Array of voltages to compute tau for.
% T:        Temperature in degrees Celcius. 
% 
% Output:
% tau:      Array containing the time constants at each input voltage. 
function tau = calc_tau (alpha, beta, V, T)
    k   = 3^(0.1 * (T - 6.3));
    tau = 1 ./ (k *(alpha(V) + beta(V)));
end