# ====================================================================================================
# Class that to simulate the cooperation model using the method of lines.
# ====================================================================================================
# Import libraries
import numpy as np
import scipy.integrate
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class fullModelSolver():
    def __init__(self, paramsList=[1.2, .5, .5, 1, 4e-5, 70, 12.5, 1, 1e-4],
                 domainSizeList=[0, 1, 25],
                 dt=0.1, dx=0.01):

        # Parameterise Equation
        self.cS = paramsList[0]  # Inhibition Stroma -> Tumour
        self.cMA = paramsList[1]  # Inhibition T_M -> T_A
        self.cAM = paramsList[2]  # Inhibition T_A -> T_E
        self.rT = paramsList[3]  # Relationship stroma and tumour growth rates
        self.DT = paramsList[4]  # Mobility of Tumour (in reference to acid movement)
        self.rL = paramsList[5]  # Stiffness of acid equation
        self.dS = paramsList[6]  # Stroma degradation by acid
        self.dM = paramsList[7]  # Matrix degradation rate
        self.eps = paramsList[8]  # Leakage of tumour cells through completely in tact matrix (0 for no leakage)

        # Define the domain
        self.xMin = domainSizeList[0]
        self.xMax = domainSizeList[1]
        self.tEnd = domainSizeList[2]

        # Set the parameters for the ICs to the default. User can change them
        # manually with Set_ICParameters() later.
        self.Set_ICParameters()
        self.Initialised = False  # Keep track of solution matrices have been initialised.

        # Define discretisation
        self.dt = dt  # Maximum allowed time step in the adaptive solver
        self.dx = dx  # Spatial Mesh Size

        # Compute the number of grid points
        self.nTimeSteps = int(math.floor(self.tEnd / self.dt) + 1)
        self.nGrdPts = int(math.floor((self.xMax - self.xMin) / self.dx) + 1)
        self.xVec = np.linspace(self.xMin, self.xMax, self.nGrdPts)
        self.loggedTimeVec = np.linspace(0, self.tEnd, self.nTimeSteps)

        # Helper variables
        self.SolvedB = False  # Indicate that solver hasn't been Run

    # ============================================================
    # Function to parameterise the ICs. This sets the position of the boundaries
    # of each population/molecule at the beginning of the simulation.
    # Format: [x-position of edge, width of edge (how quickly it decays from 0 to 1)]
    def Set_ICParameters(self, stromaParamList=[0.2, 0.2], tumourAParamList=[0.2, 0.2],
                         tumourMParamList=[0.2, 0.2], matrixParamList=[0.2, 0.2],
                         acidParamList=[0.2, 0.2]):
        self.stromaParamList = stromaParamList
        self.tumourAParamList = tumourAParamList
        self.tumourMParamList = tumourMParamList
        self.matrixParamList = matrixParamList
        self.acidParamList = acidParamList

    # Initial conditions
    def MollifiedStep(self, x, edge=0.3, width=0.2):
        y = np.zeros_like(x)
        y[x < (edge - width)] = 1.
        y[(x >= (edge - width)) & (x < edge)] = np.exp(
            -1 / (1 - np.square(np.abs(x[(x >= (edge - width)) & (x < edge)] - (edge - width)) / width))) * np.exp(1)
        return y

    def IC_stroma(self, x):
        return 1 - self.MollifiedStep(x, self.stromaParamList[0], self.stromaParamList[1])

    def IC_tumourA(self, x):
        return self.MollifiedStep(x, self.tumourAParamList[0], self.tumourAParamList[1])

    def IC_tumourM(self, x):
        return self.MollifiedStep(x, self.tumourMParamList[0], self.tumourMParamList[1])

    def IC_matrix(self, x):
        return 1 - self.MollifiedStep(x, self.matrixParamList[0], self.matrixParamList[1])

    def IC_acid(self, x):
        return self.MollifiedStep(x, self.acidParamList[0], self.acidParamList[1])

    # ============================================================
    # Function to initialise the matrices to hold the solutions
    def ApplyInitialConditions(self):
        self.S = np.reshape(self.IC_stroma(self.xVec), (self.nGrdPts, 1))
        self.TA = np.reshape(self.IC_tumourA(self.xVec), (self.nGrdPts, 1))
        self.TM = np.reshape(self.IC_tumourM(self.xVec), (self.nGrdPts, 1))
        self.M = np.reshape(self.IC_matrix(self.xVec), (self.nGrdPts, 1))
        self.A = np.reshape(self.IC_acid(self.xVec), (self.nGrdPts, 1))
        self.Initialised = True

    # ============================================================
    # Main simulation function
    # nLogPts - interval at which system states are saved. NOTE: Will always save the
    # first and last state, and then nLogPts-2/logInterval states in between.
    def Run(self, logInterval=None, method='BDF', rtol=1e-6, atol=1e-9):
        self.rTol = rtol
        self.aTol = atol
        # ---------------------------------------------------
        # If user decides to log solution at other intervals than dt
        if logInterval != None:
            self.nLogPts = math.floor(self.tEnd / logInterval) + 1
            self.loggedTimeVec = np.linspace(0, self.tEnd, self.nLogPts)

        # Initialise the solutions
        if not self.Initialised: self.ApplyInitialConditions()

        # Solve
        initialStateVec = np.concatenate([self.S[:, 0], self.TA[:, 0], self.TM[:, 0], self.A[:, 0], self.M[:, 0]])
        self.solverObj = scipy.integrate.solve_ivp(self.SpatiallyDiscretisedSystem,
                                                   t_span=(0, self.tEnd), y0=initialStateVec,
                                                   method=method, max_step=self.dt,
                                                   t_eval=self.loggedTimeVec,
                                                   rtol=self.rTol, atol=self.aTol)
        self.SolvedB = True
        self.S, self.TA, self.TM, self.A, self.M = [self.solverObj.y[k * self.nGrdPts:(k + 1) * self.nGrdPts] for k in
                                                    range(5)]

        # ============================================================

    # By discretising in space we can derive the following ODE system, which we solve
    # in time using a RK scheme. The ODEs for the different componenents (S,TA,TM,A,M)
    # are arranged in one long vector in that order.
    def SpatiallyDiscretisedSystem(self, t, uVec):
        currSVec, currTAVec, currTMVec, currAVec, currMVec = [uVec[k * self.nGrdPts:(k + 1) * self.nGrdPts] for k in
                                                              range(5)]
        # Spatial ODEs for the stroma, S
        dudtVec_S = self.f_stroma(currSVec, currAVec)
        # ODEs from spatially discretised PDE for TA
        dudtVec_TA = self.f_tumourA(currSVec, currTAVec, currTMVec) + self.InhomogeneousDiffusionOperator(currTAVec,
                                                                                                          self.DT * (
                                                                                                                      1 - currMVec / (
                                                                                                                          1 + self.eps)))
        # ODEs from spatially discretised PDE for TM
        dudtVec_TM = self.f_tumourM(currSVec, currTAVec, currTMVec) + self.InhomogeneousDiffusionOperator(currTMVec,
                                                                                                          self.DT * (
                                                                                                                      1 - currMVec / (
                                                                                                                          1 + self.eps)))
        # ODEs from spatially discretised PDE for A
        dudtVec_A = self.f_acid(currTAVec, currAVec) + self.InhomogeneousDiffusionOperator(currAVec,
                                                                                           np.ones_like(currAVec))
        # Spatial  ODEs for the matrix, M
        dudtVec_M = self.f_matrix(currTMVec, currMVec)

        return np.concatenate([dudtVec_S, dudtVec_TA, dudtVec_TM, dudtVec_A, dudtVec_M])

    # ============================================================
    # Source terms of the different equations
    # RHS of the stroma ODE
    def f_stroma(self, S, A):
        return S * (1 - S) - self.dS * A * S

    # Source terms of the acid producing tumour PDE
    def f_tumourA(self, S, TA, TM):
        return self.rT * TA * (1 - self.cS * S - TA - self.cMA * TM)

    # Source terms of the matrix degrading tumour PDE
    def f_tumourM(self, S, TA, TM):
        return self.rT * TM * (1 - self.cS * S - self.cAM * TA - TM)

    # Source terms of the acid PDE
    def f_acid(self, TA, A):
        return self.rL * (TA - A)

    # Source terms of the matrix ODE
    def f_matrix(self, TM, M):
        return -self.dM * TM * M

    # ============================================================
    # Define the spatial discretisation for the diffusion operator
    def InhomogeneousDiffusionOperator(self, UVec, DVec):
        dudxVec = np.zeros_like(UVec)
        dudxVec[1:-1] = ((DVec[:-2] + DVec[1:-1]) * UVec[:-2] - (DVec[:-2] + 2 * DVec[1:-1] + DVec[2:]) * UVec[1:-1] + (
                    DVec[1:-1] + DVec[2:]) * UVec[2:]) / (2 * self.dx ** 2)
        dudxVec[0] = ((DVec[1] + DVec[0]) * UVec[1] - (DVec[0] + DVec[1]) * UVec[0]) / self.dx ** 2
        dudxVec[-1] = ((DVec[-2] + DVec[-1]) * UVec[-2] - (DVec[-2] + DVec[-1]) * UVec[-1]) / self.dx ** 2
        return dudxVec

    # ============================================================
    # Function to plot the state of the system at a set of points in time
    def Plot(self, timesToPlotList, saveFigB=False, figsize=None, lineWidth=3, ylimVec=[0, 1.1], outName=None):
        tIdxList = [np.argwhere(np.abs(self.loggedTimeVec - tVal) < (self.dx / 2))[0][0] for tVal in timesToPlotList]
        nPlots = len(tIdxList)
        if figsize != None: plt.figure(figsize=figsize)
        for pltIdx in range(nPlots):
            tIdx = tIdxList[pltIdx]
            plt.subplot("%i%i%i" % (nPlots, 1, pltIdx + 1))
            plt.fill_between(self.xVec, np.zeros_like(self.M[:, tIdx]), self.M[:, tIdx],
                             color='#B9B9BA', lw=lineWidth, alpha=0.6, hatch='..',
                             label="ECM")
            plt.fill_between(self.xVec, np.zeros_like(self.S[:, tIdx]), self.S[:, tIdx],
                             color='#8079FE', lw=lineWidth, alpha=0.6, hatch='//',
                             label="Stroma")
            plt.fill_between(self.xVec, np.zeros_like(self.A[:, tIdx]), self.A[:, tIdx],
                             color='#B676AA', lw=lineWidth, alpha=0.5, hatch='**',
                             label="Acid")
            plt.fill_between(self.xVec, np.zeros_like(self.TA[:, tIdx]), self.TA[:, tIdx],
                             color='#FD676B', lw=lineWidth, alpha=0.6, hatch='xx',
                             label=r"$T_A$")
            plt.fill_between(self.xVec, np.zeros_like(self.TM[:, tIdx]), self.TM[:, tIdx],
                             color='#DDDA6D', lw=lineWidth, alpha=0.6, hatch='++',
                             label=r"$T_M$")
            plt.xlabel("x")
            plt.ylabel("Non-Dimensionalised \n Variables")
            plt.title(r"Time t = %1.2f" % self.loggedTimeVec[tIdx])
            plt.ylim(ylimVec)
            if pltIdx == 0:
                legend = plt.legend(loc='upper right', shadow=False,
                                    frameon=True, framealpha=1, facecolor=(1, 1, 1),
                                    prop={'size': 22})
        plt.tight_layout()
        # Save the figure if requested
        if saveFigB:
            if outName == None:
                outName = "t_%1.2f_dt_%1.3f_dx_%1.3f.pdf" % (self.timesToPlotList[tIdx], self.dt, self.dx)
            plt.savefig(outName, orientation='portrait', format='pdf')
            plt.close()