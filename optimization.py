import numpy as np
from scipy.optimize import linprog
import pandas as pd



# TODO: indifference thresholds, checking for 0 division, no "lower is better", add assertions
def normalize(gc_data):
    return (gc_data - gc_data.min()) / (gc_data.max() - gc_data.min())

# sensitivity analysis with this normalization, not per RCP.
# iland_one_gc_normalized = (iland_one_gc - iland_one_gc.min()) / (iland_one_gc.max() - iland_one_gc.min())

def convert_to_optimizer_input(gc_data):
    df_long = gc_data.stack().rename("value")
    df_long.index.set_names(['rid', 'Germany_id', "RCPScenario", "management", "ES"], inplace=True)
    df_long = df_long.reorder_levels(["ES", 'Germany_id', 'rid', "RCPScenario", "management"])
    df_long = df_long.unstack("management")
    df_long.columns.name = None # removes an arbitrarily added "management" column, the managements are their own columns now.
    return df_long


TOLERANCE = 0.00001


def prepare_for_optimization(gc_data):
    normalized_gc_data = gc_data.groupby('RCPScenario', group_keys=False).apply(normalize)

    return convert_to_optimizer_input(normalized_gc_data)


def prepare_and_optimize_gridcell(gc_data):
    optimizer_data = prepare_for_optimization(gc_data)
    return optimize_gridcell(optimizer_data, gc_data.index.get_level_values('rid')[0])


def optimize_gridcell(gc_data_for_optimizer, rid, es_weights=None):

    opt_result = solve_optimization_for_gridcell_general_min_max_distance(gc_data_for_optimizer.values, ['rcp26', 'rcp45', 'rcp85'], es_weights=es_weights)

    assert opt_result.success == True, "Optimization failed for gc " + str(rid)

    portfolio = opt_result.x[1:]
    portfolio[np.abs(portfolio) < TOLERANCE] = 0.0

    portfolio_sum = np.sum(portfolio)
    assert np.abs(portfolio_sum - 1) < TOLERANCE, "Optimization result does not sum to 1 but " + str(portfolio_sum) + " for gc " + str(rid)
    assert np.all(portfolio >= 0), "negative portfolio values"

    managements = list(gc_data_for_optimizer.columns)
    return pd.Series({managements[i]: portfolio[i] for i in range(len(managements))})

def compute_mean_portfolios(grp):
    hexagon_mean_portfolio = grp.mean(axis=0)
    portfolio_sum = hexagon_mean_portfolio.values.sum()
    assert np.abs(portfolio_sum-1) < TOLERANCE, "Averaged portfolio does not sum to 1 but to " + str(portfolio_sum) + " for hexagon " + str(grp.index.get_level_values('Germany_id')[0])
    return hexagon_mean_portfolio



def solve_optimization_for_gridcell_general_min_max_distance(es_vals, rcps, es_weights=None, m_upper_bounds=None, m_lower_bounds=None, additional_constraints=None, lambda_opt=0, infeasible_management_idxs=[]):
    obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq, bounds = get_optimization_inputs_for_gridcell_general_min_max_distance(es_vals, rcps, es_weights, m_upper_bounds, m_lower_bounds, additional_constraints, lambda_opt, infeasible_management_idxs)

    return linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                   A_eq=lhs_eq, b_eq=rhs_eq, bounds=bounds,
                   method="revised simplex")

# converts all the ES values etc into the proper mathematical values, arrays, etc., to be then plugged into the solver
def get_optimization_inputs_for_gridcell_general_min_max_distance(es_vals, rcps, es_weights=None, m_upper_bounds=None, m_lower_bounds=None, additional_constraints=None, lambda_opt=0, infeasible_management_idxs=[]):

    """
    This assumes that the es_vals are already normalized.
    In order to be able to use weights, it is easier to optimize the maximal distance to the optimum instead of minimizing the worst case.
    The result is the same, but including the weights is mathematically more convenient.
    :param es_vals: the value matrix of dimension (#es, #strategies)
    :param rcps: just used to double check the size of es_vals
    :param es_weights: should be a list of doubles with dimension (1, #es) to give a weighting for each ecosystem indicator.
                        the weightings do not need to sum up to 1 as by construction, the es_vals are bound to [0, 1] so the weighting does not need to be normalized, the result will be the same.
    :param m_upper_bounds: upper bounds for the distribution of management strategies. Here can be specified how large a fraction of the gridcell may maximally be given to one strategy.
    :param m_lower_bounds: lower bounds for the management strategies.
    :param additional_constraints: dictionary with LHS and RHS of additional constraints in the form LHS >= RHS, e.g. lower bound for c sequestration
    :return: solution of the linear program, including meta information. Get the distribution via solution.x[1:] (the first value of x is a value of a proxy variable)
    """

    n_vals = np.squeeze((es_vals.shape[0],)) / len(rcps)

    # e.g: z, omega_base, omega_toBd, omega_toNe, omega_toFree, omega_toFreeB
    obj = np.zeros(es_vals.shape[1] + 1)
    obj[0] = 1

    dists = np.ones_like(es_vals) - es_vals

    # with a control parameter lambda to get a solution between balanced and average (see diaz-balteiro 2018)
    # see methods section of optimization paper.

    if es_weights is None:
        es_weights = np.ones_like(es_vals) / es_vals.shape[0]
    else:
        n_weights = np.squeeze(np.shape(es_weights))
        assert n_weights == n_vals, "not the right dimension of weights!"

        if len(rcps) > 1:
            # [0.5, 0.2, 0.3] --> [0.5, 0.5, 0.2, 0.2, 0.3, 0.3]
            es_weights = np.repeat(es_weights, len(rcps))

        es_weights = np.transpose(np.tile(np.array(es_weights), (es_vals.shape[1], 1)))

    if lambda_opt > 0:
        obj[0] = 1-lambda_opt
        obj[1:] = np.dot(np.transpose(es_weights[:, 0]), dists) * lambda_opt/dists.shape[0]

    # one inequality per indicator with -1 for the z and k values for the k strategies
    # e.g. sum_s omega_s dist(es, s) - z <= 0
    lhs_ineq = np.ones((es_vals.shape[0], es_vals.shape[1] + 1)) * (-1)
    lhs_ineq[:, 1:] = es_weights * dists
    rhs_ineq = np.zeros_like(es_weights[:, 0])

    rows_all_zero, = np.where(np.count_nonzero(es_vals, axis=1) == 0)  # 1 because the z variable will always have coefficient 1
    if len(rows_all_zero) > 0:
        print('Warning: ' + str(len(rows_all_zero)) + ' rows where found where an indicator for an RCP had all zeros for each strategy.')

    if additional_constraints is not None:
        # additional inequalities
        assert additional_constraints['lhs'].shape[0] == len(additional_constraints['rhs']) or\
               (len(additional_constraints['lhs'].shape) == 1 and len(additional_constraints['rhs']) == 1), "shape mismatch, lhs=" + str(additional_constraints['lhs']) + " vs rhs=" + str(additional_constraints['rhs'])

        # one 0 for the z, k values for the k strategies. In the other dimension, one line per rcp.
        lhs_ineq_add = np.zeros((len(additional_constraints['rhs']), es_vals.shape[1] + 1))
        lhs_ineq_add[:, 1:] = np.array(additional_constraints['lhs']) * (-1)
        rhs_ineq_add = np.array(additional_constraints['rhs']) * (-1)
        lhs_ineq = np.concatenate((lhs_ineq, lhs_ineq_add))
        rhs_ineq = np.concatenate((rhs_ineq, rhs_ineq_add))

    # one equality for the sum of management fractions to be 1
    lhs_eq = [list(np.ones(es_vals.shape[1] + 1))]
    lhs_eq[0][0] = 0
    rhs_eq = [1]

    # one equality for every infeasible management w_man = 0
    for inf_man in infeasible_management_idxs:
        print('Warning, management ' + str(inf_man) + ' is infeasible!')
        lhs_eq_inf = [0 for i in range(es_vals.shape[1] + 1)]
        lhs_eq_inf[inf_man+1] = 1  # first column is the z value, so need to add 1!
        lhs_eq.append(lhs_eq_inf)
        rhs_eq.append(0)

    if m_upper_bounds is None:
        m_upper_bounds = np.ones(es_vals.shape[1])
    else:
        assert m_upper_bounds.shape == (es_vals.shape[1],)
    if m_lower_bounds is None:
        m_lower_bounds = np.zeros(es_vals.shape[1])
    else:
        assert m_lower_bounds.shape == (es_vals.shape[1],)

    assert np.all(m_lower_bounds < m_upper_bounds), "The provided lower bounds were not all smaller than the provided upper bounds. This is an infeasible problem."

    # first bound is for the proxy variable z
    bounds = [(0, float("inf"))] + [(m_lower_bounds[x], m_upper_bounds[x]) for x in range(es_vals.shape[1])]

    return obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq, bounds