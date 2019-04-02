from .canonical_linear import CanonicalLinearConstraint, canlinear_colloc_to_interpolate
from .constraint import DiscretizationType
import numpy as np
from toppra.constants import MAXSD

class CanonicalLinearFirstOrderConstraint(CanonicalLinearConstraint):
    """A class to represent Canonical Linear Generalized First-order constraints.

    Parameters
    ----------
    inv_dyn: (array, array, array) -> array
        The "inverse dynamics" function that receives joint position and velocity
        as inputs and ouputs constraint value. See notes for more
        details.
    cnst_F: array -> array
        Coefficient function. See notes for more details.
    cnst_g: array -> array
        Coefficient function. See notes for more details.
    dof: int, optional
        Dimension of joint position vectors. Required.

    Notes
    -----
    A First Order Constraint can be given by the following formula:

    .. math::
        A(q) \dot q = w,

    where w is a vector that satisfies the polyhedral constraint:

    .. math::
        F(q) w \\leq g(q).

    Notice that `inv_dyn(q, qd, qdd) = w` and that `cnsf_coeffs(q) =
    F(q), g(q)`.

    To evaluate the constraint on a geometric path `p(s)`, multiple
    calls to `inv_dyn` and `const_coeff` are made. Specifically one
    can derive the first-order equation as follows

    .. math::
        A(q) p'(s) \dot s = w,
        a(s) \dot s = w

    To evaluate the coefficients a(s), inv_dyn is called
    repeatedly with appropriate arguments.
    """
    def __init__(self, inv_dyn, cnst_F, cnst_g, dof, discretization_scheme=DiscretizationType.Interpolation):
        super(CanonicalLinearFirstOrderConstraint, self).__init__()
        self.set_discretization_type(discretization_scheme)
        self.inv_dyn = inv_dyn
       	self.cnst_F = cnst_F
        self.cnst_g = cnst_g
        self.dof = dof
        self._format_string = "    Kind: Generalized First-order constraint\n"
        self._format_string = "    Dimension:\n"
        F_ = cnst_F(np.zeros(dof))
        self._format_string += "        F in R^({:d}, {:d})\n".format(*F_.shape)

    def compute_constraint_params(self, path, gridpoints, scaling):
		assert path.get_dof() == self.get_dof(), ("Wrong dimension: constraint dof ({:d}) "
                                                  "not equal to path dof ({:d})".format(
                                                      self.get_dof(), path.get_dof()))
		p = path.eval(gridpoints / scaling)
		ps = path.evald(gridpoints / scaling) / scaling
		N = gridpoints.shape[0] - 1
		xbound = np.zeros((N + 1, 2))
		for i in range(N + 1):
			sdmin = - MAXSD
			sdmax = MAXSD
			F = self.cnst_F(p[i,:])
			g = self.cnst_g(p[i,:])
			a = F.dot(self.inv_dyn(p[i,:], ps[i,:]))
			for k in range(g.shape[0]):
				if a[k] > 0:
					sdmax = min(g[k] / a[k], sdmax)
				elif a[k] < 0:
					sdmin = max(g[k] / a[k], sdmin)
			xbound[i, 1] = sdmax**2
			xbound[i, 0] = max(sdmin, 0.)**2
		return None, None, None, None, None, None, xbound
