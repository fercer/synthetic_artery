import numpy as np

class BSpline():
    def __init__(self, knots=3):
        self._knots = knots


    def _spline_basis(self, u, i, k, t_nodes):
        if k == 1:
            # Spline of degree 0:
            if t_nodes[i] <= u and u < t_nodes[i+1]:
                return 1
            
            else:
                return 0
            
        else:
            num_1 = u - t_nodes[i]
            den_1 = t_nodes[i+k-1] - t_nodes[i]
                            
            num_2 = t_nodes[i+k] - u
            den_2 = t_nodes[i+k] - t_nodes[i+1]

            if den_1 < 1e-8:
                coef_1 = 0
            else:
                coef_1 = num_1/den_1
                            
            if den_2 < 1e-8:
                coef_2 = 0
            else:
                coef_2 = num_2/den_2

            return coef_1 * self._spline_basis(u, i, k-1, t_nodes) + coef_2 * self._spline_basis(u, i+1, k-1, t_nodes)
        

    def fit(self, P, n_points=100):
        n, d = P.shape
        m = self._knots + n + 1
        t_knots = np.array([0.]*self._knots + list(range(1,n-self._knots+1)) + [n-self._knots+1]*self._knots)/float(n-self._knots+1)
        
        x = np.linspace(0., 1.-1e-12, n_points)
        C_bspline = np.zeros([n_points, d])
        
        for i, x_i in enumerate(x):
            C_bspline[i,:] = np.dot(P.transpose(), [self._spline_basis(x_i, j, self._knots, t_knots) for j in range(n)])
            
        return C_bspline




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    bspline_fitter = BSpline(knots=3)

    x = np.linspace(-np.pi, np.pi, 50)
    y = 0.0*x
    y[4] = 1.0

    P = np.column_stack((x, y))

    plt.plot(x, y, label='original')
    bspline_fitting = bspline_fitter.fit(P, n_points=100)

    print('bspline shape: ', bspline_fitting.shape)
    plt.plot(bspline_fitting[:,0], bspline_fitting[:,1], 'g-x', label='bezier')
    
    plt.legend()
    plt.show()