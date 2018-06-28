import numpy

def get_cell_1_loss_weight(targets, sizes=None, sqrt_sizes=False):

    assert sizes is not None
    return sizes.astype('float')**0.5
    # get certainty from targets
    # # => how far are we away from 0.5
    # # [0,1]
    # certainty = 4.0 * (targets - 0.5)**2

    # # compute class frequencies
    # n_targets = len(targets)


    # prio0 = (targets[numpy.where(targets >  0.5)]).sum()
    # prio1 = (targets[numpy.where(targets <= 0.5)]).sum()
    # normalization = prio0 + prio1
    # prio0 = prio0 / normalization
    # prio1 = prio1 / normalization

    

    
    


    # #prio0 = (targets<0.5).sum() / n_targets
    # #prio1 = 1.0 - prio0

    # prio0 = numpy.clip(prio0, 0.0001,0.9999)
    # prio1 = numpy.clip(prio1, 0.0001,0.9999)

    # reciprocal_prio0 = 1.0 / prio0
    # reciprocal_prio1 = 1.0 / prio1


    # # from the fact that whe have unbalanced class set
    # weight_target    = (1.0 - targets)*reciprocal_prio0 + (targets*reciprocal_prio1)*12.0
    
    # weight_certainty = (certainty + 1.0)**2

    # if sizes is not None:
    #     # total weight
    #     weight_sizes     = numpy.require(sizes, dtype='float')
    #     if sqrt_sizes:
    #         weight_sizes = numpy.sqrt(weight_sizes) 
    #     ret = weight_target * weight_sizes * weight_certainty
    # else:
    #     ret = weight_target * weight_certainty

    # return numpy.require(ret, dtype='float32')


def make_cell_0_gt(cell_bounds, cell_1_gt, jsize):
    fuzzy_gt = cell_1_gt[cell_bounds-1]
    hard_gt  = numpy.round(fuzzy_gt).astype('int')
    # [0,1]
    certainty = 1.0 - numpy.sum(numpy.abs(fuzzy_gt - hard_gt), axis=1)/float(jsize)

    if  jsize == 3:
        mapping = -1*numpy.ones([2,2,2], dtype='int')
        l = 0
        for x0 in range(2):
            for x1 in range(2):
                for x2 in range(2):

                    s = x0 + x1 + x2 
                    if s != 1 :
                        mapping[x0, x1, x2] = l
                        l += 1
        mapped_gt = mapping[hard_gt[:,0],hard_gt[:,1],hard_gt[:,2]]
    elif jsize ==4:
        mapping = -1*numpy.ones([2,2,2,2], dtype='int')
        l = 0
        for x0 in range(2):
            for x1 in range(2):
                for x2 in range(2):
                    for x3 in range(2):

                        s = x0 + x1 + x2 + x3
                        if s != 1 :
                            mapping[x0, x1, x2, x3] = l
                            l += 1
        mapped_gt = mapping[hard_gt[:,0],hard_gt[:,1],hard_gt[:,2],hard_gt[:,3]]
    else:
        assert False

    #print("mapped gt shape",mapped_gt.shape)
    where_wrong = numpy.where(mapped_gt==-1)[0]
    mapped_gt[where_wrong] = 0

    class_prio = numpy.bincount(mapped_gt)/len(mapped_gt)
    class_prio = numpy.clip(class_prio, 0.00001, 1.0)
    reciprocal_class_prio = 1.0/class_prio

    weight_from_class_freq = reciprocal_class_prio[mapped_gt]


    certainty[where_wrong] = 0.0
    w = numpy.require(certainty*weight_from_class_freq, dtype='float32')
    return mapped_gt,w


