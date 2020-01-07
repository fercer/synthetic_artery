import numpy as np
from scipy.optimize import fsolve


class EllipsoidIntersection():
    def __init__(self, a = 1., c = 2., segment_points_density=500):
        self._a = a
        self._c = c
                
        self._ap = a
        self._cp = a
        
        self._u_2 = np.array((0., 1., 0.))
        self._u_1 = np.array((1., 0., 0.))

        self._p_cp = np.array((0., 0., 0.))
        self._c_e = np.array((0., 0., 0.))

        self._arc_n_points = segment_points_density
        self._arc_length = None
    
    
    def _compute_theta(self, position):      
        A_mat = np.hstack((self._ap*self._u_1[..., np.newaxis], self._cp*self._u_2[..., np.newaxis]))
        selected_rows = np.argsort(np.sum(np.abs(A_mat), axis=1))[1:]
        A_mat = np.linalg.inv(A_mat[selected_rows, :])
        
        theta = np.arccos(np.dot(A_mat, position[selected_rows])[0])
        recovered_position = self.get_position(theta)[0]
        if not np.all(np.abs(recovered_position - position) < 1e-6):
            theta = 2.*np.pi - theta
        
        return theta
    

    def _compute_arc_length(self, theta_src, theta_dst, precision=1e-3):
        theta = np.arange(theta_src, theta_dst, precision * (1 - 2*(theta_src > theta_dst)))

        ### Evaluate the circunference on the given angles:
        circunference = np.sqrt((self._ap*np.cos(theta))**2 + (self._cp*np.sin(theta)**2))
        
        ### Compute the length as the integral of the radii along the angle between theta 1 and 2
        return np.sum(circunference[1:] + circunference[:-1]) / 2. * precision

        
    def compute_intersection(self, intersection_plane, d=0.):
        m, n, p = intersection_plane
        
        p_cp = d/(m**2+n**2+p**2) * np.array([m, n, p])
        u_2 = np.array([-p*m, -p*n, m**2+n**2]) / np.sqrt((m**2+n**2)*(m**2+n**2+p**2))
        
        # Compute the scalars that give the intersection of the intersection ellipse with the plane
        ta = (u_2[0]**2+u_2[1]**2)/self._a**2 + u_2[2]**2/self._c**2
        tb = 2.*((u_2[0]*p_cp[0] + u_2[1]*p_cp[1])/self._a**2 + u_2[2]*p_cp[2]/self._c**2)
        tc = (p_cp[0]**2+p_cp[1]**2)/self._a**2 + p_cp[2]**2/self._c**2 - 1.
        
        t_discriminant = tb**2 - 4*ta*tc
        if t_discriminant < 1e-8:
            print('Warning: there no exists intersection of the ellipsoid with the plane provided')
            self._ap = 0.
            self._cp = 0.
                                
            self._u_2 = np.array((0., 0., 0.))
            self._u_1 = np.array((0., 0., 0.))

            self._p_cp = np.array((0., 0., 0.))
            self._c_e = np.array((0., 0., 0.))

            
        t2_1 = (-tb - np.sqrt(t_discriminant)) / (2*ta)
        t2_2 = (-tb + np.sqrt(t_discriminant)) / (2*ta)
        
        u_t2_1 = t2_1 * u_2 + p_cp
        u_t2_2 = t2_2 * u_2 + p_cp
        c_e = (u_t2_1 + u_t2_2) / 2.
        
        u_1 = np.array([n, -m, 0.]) / np.sqrt(m**2+n**2)

        ta = (u_1[0]**2+u_1[1]**2)/self._a**2 + u_1[2]**2/self._c**2
        tb = 2.*((u_1[0]*c_e[0] + u_1[1]*c_e[1])/self._a**2 + u_1[2]*c_e[2]/self._c**2)
        tc = (c_e[0]**2+c_e[1]**2)/self._a**2 + c_e[2]**2/self._c**2 - 1.
        
        t_discriminant = tb**2 - 4*ta*tc
        
        t1_1 = (-tb - np.sqrt(t_discriminant)) / (2*ta)
        t1_2 = (-tb + np.sqrt(t_discriminant)) / (2*ta)
        
        ap = np.max([t1_1, t1_2])
        cp = np.sqrt(np.dot(u_t2_2 - c_e, u_t2_2 - c_e))
        
        self._u_1 = u_1
        self._u_2 = u_2
        self._c_e = c_e
        self._ap = ap
        self._cp = cp
        
        
    def compute_plane_arc(self, src, dst, precision=1e-3):
        circunference = self._compute_arc_length(0.0, 2.*np.pi, precision)
        test_theta_src = self._compute_theta(src - self._c_e)
        test_theta_dst = self._compute_theta(dst - self._c_e)
        A_arc_length = self._compute_arc_length(test_theta_src, test_theta_dst, precision)
        B_arc_length = circunference - A_arc_length
            
        if A_arc_length < B_arc_length:
            self._theta_src = test_theta_src
            self._theta_dst = test_theta_dst
            self._arc_length = A_arc_length

        else:
            if test_theta_src > test_theta_dst:
                self._theta_src = test_theta_src - 2.0*np.pi

            else:
                self._theta_src = test_theta_src + 2.0*np.pi

            self._theta_dst = test_theta_dst
            self._arc_length = B_arc_length

        self._arc_n_points = int(np.max((self._arc_n_points * self._arc_length / circunference, 2)))


    def get_arc(self):
        theta = np.linspace(self._theta_src, self._theta_dst, self._arc_n_points)
        if self._theta_src < 0.0:
            print(theta)

        return self.get_position(theta)

    
    def get_position(self, theta):
        position = self._c_e[...,np.newaxis] + self._u_1[...,np.newaxis] * np.cos(theta)[np.newaxis] * self._ap + self._u_2[...,np.newaxis] * np.sin(theta)[np.newaxis] * self._cp
        return position.T
    
    
    def get_vectors(self):
        return self._u_1, self._u_2, self._c_e
        
    
    def get_theta(self, position):
        return self._compute_theta(position)
    
    
    def get_pars(self):
        return self._ap, self._cp


    def get_arc_lenth(self):
        return self._arc_length


    def compute_angles(self, position):
        phi = np.arcsin(position[2]/self._c)
        theta = np.arctan(position[1]/position[0])

        rec_position = self.get_position_ellipsoid(theta, phi)
        if np.sum(rec_position[:2] * position[:2]) < 0.0:
            theta += np.pi

        return theta, phi


    def get_position_ellipsoid(self, theta, phi):
        cphi = np.cos(phi)

        pos_x = self._a * np.cos(theta) * cphi
        pos_y = self._a * np.sin(theta) * cphi
        pos_z = self._c * np.sin(phi)

        return np.array((pos_x, pos_y, pos_z))