data {
    int<lower=0> Noff; // Number of flies in "off at X" group
    vector<lower=0>[Noff] termoff; // Termination time of flies in "off at X" group
    vector<lower=0>[Noff] lightoff; // Time of light off
    int<lower=0> Naccum; // Number of flies in accum sample 
    vector<lower=0>[Naccum] termaccum; // termination time of flies in accum sample
}
parameters {
    real<lower=0> mu; // Average time from switch to termination
    real<lower=0> sig; // Std of time from switch to termination
    real<lower=10> tauaccum; // Time of switch in the accumulated condition
}
model {
    termoff ~ normal(mu+lightoff+1.0, sig);
    termaccum ~ normal(mu+tauaccum, sig);
}