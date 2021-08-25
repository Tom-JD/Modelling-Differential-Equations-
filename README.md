# Modelling-Differential-Equations-
Coursework from Warwick Maths module MA261 - Differential Equations: Modelling and Numerics. Achieved a total grade of 90%.

Please note: part 1, 2, 3 and 4 are all branches of main.

Within this module, we investigated different methods to approximating the solutions to ordinary differential equations. These equations were formulated from a number of applications, including Hamiltonian dynamics and mass action kinetics. Methods began simple, with forward and backward Euler methods, however after investigating their convergence and stability, we attempted to improve upon them by employing discretised models. These discretised models took the form of the Runge-Kutta methods, which exist on the forefront of research to this day. Again we analysed their stability and convergence. We then proceeded to simplify our models, employing analytic methods such as perturbation, linearisation and non-dimensionalisation. The module was also littered with applications of these.

Code can be found in the second half of the four documents - some methods not included in the pdfs were supplied to us, but all code appearing in the pdfs is our own.

Part 1 is an investigation of the Forward Euler method. We implemented the method, then investigated its convergence and stability.

Part 2 is an implementation of the Backward Euler method and the Crank-Nicholson method, and an investigation into their error convergence, with a comparison to the Forward Euler method.

Part 3 firstly implements and compares error convergence of the Crank-Nicholson and Heun methods. It then employs various implemented methods from part 1 and 2 to approximate solutions to a Hamiltonian system, and compares the methods in this application.

Part 4 begins by implementing an explicit and an implicit Runge-Kutta method. They are then compared, and applied to differential equations derived from mass action kinetics. Finally we plot their performance and phase portraits of the system of equations.
