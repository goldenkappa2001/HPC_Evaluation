//runge_kunta.h
#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

/**
 * Function to perform a single step of the Runge-Kutta method.
 * Used for numerical integration in solving differential equations.
 * 
 * @param eta Current value of the variable being updated.
 * @param u Related variable used in calculations.
 * @return Updated value after applying Runge-Kutta calculations.
 */
 
double rungeKuttaStep(double eta, double u);

#endif // RUNGE_KUTTA_H
