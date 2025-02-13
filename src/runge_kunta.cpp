//runge_kunta.cpp
#include "runge_kutta.h"

double rungeKuttaStep(double eta, double u) {
    double k1 = 0.5 * eta + u;
    double k2 = 0.5 * (eta + k1) + u;
    double k3 = 0.5 * (eta + k2) + u;
    double k4 = 0.5 * (eta + k3) + u;
    return (k1 + 2*k2 + 2*k3 + k4) / 6.0;
}
