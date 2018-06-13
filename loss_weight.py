

def loss_weight(self, targets, sizes):

    # get certainty from targets
    # => how far are we away from 0.5
    # [0,1]
    certainty = 4.0 * (targets - 0.5)**2

    # compute class frequencies
    n_targets = len(targets)

    prio0 = (targets<0.5).sum() / n_targets
    prio1 = 1.0 - prio0

    reciprocal_prio0 = 1.0 / prio0
    reciprocal_prio1 = 1.0 / prio1


    # from the fact that whe have unbalanced class set
    weight_target    = (1.0 - targets)*reciprocal_prio0 + targets*reciprocal_prio1
    weight_sizes     = numpy.require(sizes, dtype='float')
    weight_certainty = (certainty + 0.1)**2

    # total weight
    return weight_target * weight_sizes * weight_certainty