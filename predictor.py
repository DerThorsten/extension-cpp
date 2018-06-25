import numpy
import torch
import vigra
import os
import h5py
import numpy
import functools
import nifty
import nifty.segmentation
import nifty.ground_truth 


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)

    return x[tuple(indices)]


def maybe_detach(array):
    if(isinstance(array, numpy.ndarray)):
        return array
    else:
        return array.cpu().detach().numpy().squeeze()


def augmentor(rgbp, masks, with_aug=True):

    rgbp_n = to_numpy(rgbp)
    masks_n = to_numpy(masks)
    print("rgbp_n",rgbp_n.shape)
    rgbp_nb = rgbp_n.copy()
    masks_nb = masks_n.copy()

    # no nothing
    yield rgbp,masks,1.0

    # # noise
    # for i in range(5):
    #     aug_rgbp = noise_augment_rgbp(rgbp, scale=0.4)
    #     yield aug_rgbp,masks, 0.05

    def fliplrud(array):
        return numpy.fliplr(numpy.flipud(array))
    def notrans(array):
        return array

    if with_aug:
        func_list = [
            numpy.fliplr,
            numpy.flipud,
            fliplrud,
            numpy.rot90
        ]


        for fuc in func_list:

            # flip lr
            for c in range(rgbp_n.shape[1]):
                rgbp_nb[0,c,:,:] = fuc(rgbp_n[0,c,:,:])
            for c in range(masks_n.shape[1]):
                masks_nb[0,c,:,:] = fuc(masks_n[0,c,:,:])

            rgbp_aug_cuda = torch.from_numpy(rgbp_nb).cuda()
            mask_aug_cuda = torch.from_numpy(masks_nb).int().cuda()
            #print("ACC")
            yield rgbp_aug_cuda,mask_aug_cuda, 1.0



        # # noise
        # for i in range(1):
        #     aug_rgbp = noise_augment_rgbp(rgbp_aug_cuda, scale=0.4)
        #     yield aug_rgbp,mask_aug_cuda, 0.05



def noise_augment_rgbp(rgbp, scale):
    print("input",rgbp.size())
    size = rgbp.size()
    to_add = numpy.random.normal(loc=0.0, scale=scale, size=size).astype('float32')
    print(to_add[0,0,0,0])
    to_add = torch.from_numpy(to_add).cuda()
    print("to_add",to_add.dtype, rgbp.dtype)
    return to_add + rgbp

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def build_backward_mapping():

    backward_mapping = numpy.zeros([5,3])
    l = 0
    for x0 in range(2):
        for x1 in range(2):
            for x2 in range(2):

                s = x0 + x1 + x2 
                if s != 1 :
                    #mapping[x0, x1, x2] = l
                    backward_mapping[l,...] = x0, x1, x2
                    l += 1

    return backward_mapping


def convert_pred(pred_c0, backward_mapping):
    preds = numpy.zeros(3)
    w = 0.0
    for l,p in enumerate(pred_c0):
        preds +=  p*backward_mapping[l,...]
        w += p
    preds/=w
    return preds

class Predictor(object):
    def __init__(self, model, ds, output_folder):
        self.model = model.eval()
        self.ds = ds
        try:
            os.stat(output_folder)
        except:
            os.mkdir(output_folder)    


        self.out_dir = os.path.join(output_folder, self.ds.split)
        try:
            os.stat(self.out_dir)
        except:
            os.mkdir(self.out_dir)    



    def predict_augmented(self, index, sp=None,tt_augment=False, with_aug=True):
        res_odict = self.ds.getitemimpl(index=index, return_test_data=True, sp=sp, tt_augment=tt_augment)
        backward_mapping = build_backward_mapping()

        #############################################
        # GET X / INPUT TENSORS
        # AND PAD THEM
        #############################################
        tlist = []
        for i,key in enumerate(res_odict.keys()):
            tlist.append(res_odict[key][None,...].cuda())
            if i + 1 == self.ds.num_inputs():
                break

        #############################################
        # DO THE AUGMENTED PREDICTION 
        #############################################

        cell1_preds_list = []
        cell0_3_preds_list = []
        lifted_preds_list = []
        weights = []


        rgbp = tlist[0]
        masks = tlist[1]
        for aug_rgbp, aug_masks, weight in augmentor(rgbp, masks, with_aug=with_aug):

            tlist[0] = aug_rgbp
            tlist[1] = aug_masks
        
            cell1_preds, cell0_3_preds, lifted_preds = self.model(
                *tlist[0:self.ds.num_inputs()]
            )
            assert cell1_preds.min()>=0.0
            assert cell1_preds.max()<=1.0
            cell1_preds_list.append(to_numpy(cell1_preds))
            cell0_3_preds_list.append(to_numpy(cell0_3_preds))
            lifted_preds_list.append(to_numpy(lifted_preds))
            weights.append(weight)

    


        cell1_preds = numpy.average(cell1_preds_list, weights=weights, axis=0).squeeze()
        cell0_3_preds = numpy.average(cell0_3_preds_list, weights=weights, axis=0).squeeze()
        lifted_preds = numpy.average(lifted_preds_list, weights=weights, axis=0).squeeze()

        assert cell1_preds.min()>=0.0
        assert cell1_preds.max()<=1.0
        
        return res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds


    def predict_sp_augmented(self, index, sp=None):
        # reference
        res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds = self.predict_augmented(index, sp=sp)
        cell_1_bounds = res_odict["cell_1_bounds"]
        lifted_edges = res_odict["lifted_edges"]


        # new lifted nh
        g = nifty.graph.undirectedGraph(cell_1_bounds.max()+1)
        g.insertEdges(cell_1_bounds)
        lifted_edges, distances = g.graphNeighbourhoodAndDistance(maxDistance=5, suppressGraphEdges=True)
        




        sp  = res_odict["sp"]

        acc_preds = cell1_preds.copy()*14.0
        acc_w = numpy.ones_like(acc_preds)*14.0

        acc_lifted_preds = numpy.zeros(lifted_edges.shape[0])
        acc_w_lifted_preds = numpy.zeros(lifted_edges.shape[0])




        for i in range(30):
            res_odict2, backward_mapping2, cell1_preds2, cell0_3_preds2, lifted_preds2 = self.predict_augmented(index, tt_augment=True,  with_aug=False)
            cell_1_bounds2 = res_odict2["cell_1_bounds"]
            lifted_edges2 = res_odict2["lifted_edges"]
            sp2  = res_odict2["sp"]


            
            assert sp.min() == 1
            assert sp2.min() == 1
            overlap = nifty.ground_truth.overlap(segmentation=sp, groundTruth=sp2)

            assert cell1_preds2.min()>=0.0
            assert cell1_preds2.max()<=1.0
            assert cell_1_bounds2.shape[0] == cell1_preds2.shape[0]


            # transfer lifted from others NON lifted
            aug_lifted_p, w_aug_lifted_p = overlap.transferCutProbabilities(
                lifted_edges,
                cell_1_bounds2,
                cell1_preds2
            )
            acc_lifted_preds += aug_lifted_p
            acc_w_lifted_preds += w_aug_lifted_p


            # transfer lifted from others non lifted
            aug_lifted_p, w_aug_lifted_p = overlap.transferCutProbabilities(
                lifted_edges,
                lifted_edges2,
                lifted_preds2
            )
            acc_lifted_preds += 0.2*aug_lifted_p
            acc_w_lifted_preds += 0.2*w_aug_lifted_p



            # transfer non lifted
            aug_cell1_preds, w_aug_preds = overlap.transferCutProbabilities(
                cell_1_bounds,
                cell_1_bounds2,
                cell1_preds2
            )
            acc_preds += aug_cell1_preds
            acc_w += w_aug_preds


        # filter out lifted edges where we have no
        # meassurements / predictions
        where_valid = numpy.where(acc_w_lifted_preds>0.0001)[0]
        acc_lifted_preds = acc_lifted_preds[where_valid]
        acc_w_lifted_preds = acc_w_lifted_preds[where_valid]
        lifted_edges = lifted_edges[where_valid,:]
        distances = distances[where_valid]

        cell1_preds_new =  acc_preds / acc_w
        lifted_preds_new = acc_lifted_preds / acc_w_lifted_preds

        #for i in range(100):
        #    print("o ",cell1_preds[i],"n ",cell1_preds_new[i],cell1_preds_new[i]-cell1_preds[i])
        #    print("oo",lifted_preds[i],"nn",lifted_preds_new[i],lifted_preds_new[i]-lifted_preds[i])
        res_odict["lifted_edges"] = lifted_edges
        res_odict["lifted_distances"] = distances
        return res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds_new


    def predict_agglo(self, index):
        import nifty.ufd

        batch = 100
        sp = None
        while(True):
            res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds = self.predict_sp_augmented(index,sp=sp)
            cell_1_bounds = to_numpy(res_odict["cell_1_bounds"])
            img_raw       = res_odict["img_raw"]
            sp            = res_odict["sp"]
            image_number  = res_odict["image_number"]
            gt_stack      = res_odict["gt_stack"]

                


            argmin = numpy.argmin(cell1_preds)
            minval = cell1_preds[argmin]
            exit = False
            cell_1_to_merge = numpy.argsort(cell1_preds)[0:batch]
            ufd = nifty.ufd.ufd(sp.max() + 1)
            print("min:",minval)
            for cell_1 in cell_1_to_merge:
                if cell1_preds[cell_1] < 0.5:
                    u = cell_1_bounds[cell_1,0]
                    v = cell_1_bounds[cell_1,1]
                    ufd.merge(u,v)
            
                else:
                    exit = True
                    break
            if exit:
                break
            labels = ufd.elementLabeling()
            sp = numpy.take(labels.astype('int'), sp.astype('int'))

        import nifty.segmentation
        import pylab
        img_small = im_raw
        ol = nifty.segmentation.segmentOverlay(image=img_small, segmentation=sp, beta=0.24)
        pylab.imshow(numpy.swapaxes(ol, 0,1))
        pylab.show()

      
    def predict_lmc(self, index):
        res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds = self.predict_sp_augmented(index)
   
        cell_1_bounds   = res_odict["cell_1_bounds"]
        sp              = res_odict["sp"]
        tgrid           = res_odict["tgrid"]
        img_raw         = res_odict["img_raw"]
        sp              = res_odict["sp"]
        image_number    = res_odict["image_number"]
        gt_stack        = res_odict["gt_stack"]

        lifted_edges     = maybe_detach(res_odict['lifted_edges'])
        lifted_distances = maybe_detach(res_odict['lifted_distances'])










        
        ############################################
        # DO STUFF WITH RESULTS!
        #############################################
        # for i in range(cell0_3_preds.shape[0]):
        #     print("C03_%d"%i, cell0_3_preds[i,...])

        # for i in range(lifted_preds.shape[0]):
        #     if(lifted_preds[i] > 0.5):
        #         print("lifted_preds%d"%i, lifted_preds[i])



        img_raw = vigra.taggedView(img_raw, "xyc")
        img_raw_big = vigra.sampling.resize(img_raw, [2*s-1 for s in img_raw.shape[0:2]])




        # bounds
        cell_bounds = tgrid.extractCellsBounds()
        cell_0_bounds = cell_bounds[0].__array__().astype('int32')
        cell_1_bounds = cell_bounds[1].__array__().astype('int32')



        cellGeometry = tgrid.extractCellsGeometry()
        cell_1_sizes  = cellGeometry[1].sizes().astype('int')
        cell_2_sizes  = cellGeometry[2].sizes().astype('int')

        cell1Geometry = cellGeometry[1]


        import numpy
        cell_1_preds_wsum = numpy.zeros(tgrid.numberOfCells[1])
        cell_1_preds_w    = numpy.zeros(tgrid.numberOfCells[1])

        # from cell 1
        for cell_1_index in range(tgrid.numberOfCells[1]):
            p1 = cell1_preds[cell_1_index]
            cell_1_preds_wsum[cell_1_index] += 1.0*p1
            cell_1_preds_w[cell_1_index] += 1.0

        # from cell 0
        if True:
            i3 = 0
            for cell_0_index in range(tgrid.numberOfCells[0]):
                bounds = cell_0_bounds[cell_0_index,:]
                if bounds[3] == 0:
                    jsize = 3
                    bounds = bounds[0:3]
                    cell_0_pred = cell0_3_preds[i3]
                    #argmx = numpy.argmax(cell_0_pred)
                    #cell_1_pred = backward_mapping[argmx, ...]
                    cell_1_pred = convert_pred(cell_0_pred, backward_mapping)
                    i3 += 1
                

                    cell_1_indices = bounds  - 1

                    for cell_1_index,p1 in zip(cell_1_indices, cell_1_pred):
                        
                        if jsize == 3:
                            cell_1_preds_wsum[cell_1_index] += 0.45*p1
                            cell_1_preds_w[cell_1_index] += 0.45
        

        cell_1_preds_combined = cell_1_preds_wsum / cell_1_preds_w

        visu = img_raw_big.copy()




        

        # from normal edges
        eps = 0.0001
        p1 = numpy.clip(cell_1_preds_combined, eps, 1.0-eps)
        p0 = 1.0 - p1 
        b_local = 0.5
        w_local = numpy.log(p0/p1) + numpy.log((1.0-b_local)/b_local)
        w_local *= cell_1_sizes.astype('float')/cell_1_sizes.max()
        w_local /= w_local.shape[0]


        # from lifted edges
        eps = 0.0001
        p1 = numpy.clip(lifted_preds, eps, 1.0-eps)
        p0 = 1.0 - p1 
        b_lifted = 0.5
        w_lifted = numpy.log(p0/p1) + numpy.log((1.0-b_lifted)/b_lifted)

        su = cell_2_sizes[lifted_edges[:,0]-1]
        sv = cell_2_sizes[lifted_edges[:,1]-1]
        ls = numpy.minimum(su,sv).astype('float')
        w_lifted *= (ls/ls.max())
        w_lifted /= w_lifted.shape[0]
        w_lifted *= 3.0
        
        w_lifted /= (lifted_distances.astype('float') + 1.0)**2




        # make a rag
        import nifty.graph





        max_node = cell_1_bounds.max()
        graph = nifty.graph.undirectedGraph(max_node + 1)
        graph.insertEdges(cell_1_bounds)

        lmc_graph = nifty.graph.undirectedGraph(max_node + 1)
        lmc_graph.insertEdges(cell_1_bounds)
        #lmc_graph.insertEdges(lifted_edges)



      

        lmc_graph_w = numpy.zeros(lmc_graph.numberOfEdges)
        #uv = rag.uvIds()


        where_neg = numpy.where(w_lifted < 0.0)[0]
        w_neg = w_lifted[where_neg]
        lifted_edges_neg = lifted_edges[where_neg,:]

        lmc_graph_w[lmc_graph.findEdges(cell_1_bounds)] = w_local
        lmc_graph_w[lmc_graph.findEdges(lifted_edges_neg)] = w_neg




        MulticutObjective = lmc_graph.__class__.MulticutObjective
        solverFactory = MulticutObjective.multicutIlpCplexFactory()
        objective = MulticutObjective(lmc_graph, lmc_graph_w)

        loggingVisitor = MulticutObjective.verboseVisitor(visitNth=1)
        solver = solverFactory.create(objective)
        result_mc = solver.optimize(loggingVisitor)

        result_mc = nifty.graph.connectedComponentsFromNodeLabels(graph, result_mc)







        obj_lmc = nifty.graph.lifted_multicut.liftedMulticutObjective(graph)
        obj_lmc.setCosts(cell_1_bounds, w_local)
        obj_lmc.setCosts(lifted_edges, w_lifted)

        # solverFactory = obj_lmc.liftedMulticutGreedyAdditiveFactory()
        # solver = solverFactory.create(obj)
        # visitor = obj.verboseVisitor(1)
        # arg = solver.optimize()

        solverFactory = obj_lmc.liftedMulticutKernighanLinFactory()
        solver = solverFactory.create(obj_lmc)
        visitor = obj_lmc.verboseVisitor(1)
        arg2 = solver.optimize(visitor,result_mc)


        pgen = obj_lmc.watershedProposalGenerator(sigma=1.0,seedingStrategie='SEED_FROM_LOCAL',
                numberOfSeeds=0.1)


        solverFactory = obj_lmc.fusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=1000,
            stopIfNoImprovement=100)

        solver = solverFactory.create(obj_lmc)
        visitor = obj_lmc.verboseVisitor(1)
        arg3 = solver.optimize(visitor, arg2)

        import pylab
        seg = numpy.take(arg3.astype('int'), sp.astype('int')).astype('uint64')













        # and again
        # make the Region adjacency graph (RAG)
        rag = nifty.graph.rag.gridRag(seg)
        gradmag = vigra.filters.gaussianGradientMagnitude(img_raw, 1.0).squeeze().view(numpy.ndarray)
        gradmag = numpy.require(gradmag, requirements=['C'])
        edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
            rag, gradmag, [10,10],1)
        meanEdgeStrength = edge_features[:,0]
        edgeSizes = edge_features[:,1]
        nodeSizes = node_features[:,1]


        minimumNodeSize = int(15**2)
        # print("minimumNodeSize",minimumNodeSize)
        # cluster-policy  
        nodeSeg = nifty.graph.agglo.sizeLimitClustering(
            graph=rag,nodeSizes=nodeSizes, edgeIndicators=meanEdgeStrength,
            edgeSizes=edgeSizes, sizeRegularizer=0.5,
            minimumNodeSize=minimumNodeSize)

        #gglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
        #gglomerativeClustering.run(verbose=0, printNth=0)
        #odeSeg = agglomerativeClustering.result()

        # convert graph segmentation
        # to pixel segmentation
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)







        img_small = vigra.sampling.resize(vigra.taggedView(img_raw,'xyc'), seg.shape)
        ol = nifty.segmentation.segmentOverlay(image=img_small, segmentation=seg, beta=0.24)
        # pylab.imshow(numpy.swapaxes(ol, 0,1))
        # pylab.show()
        #visu = img_raw.copy()
        #print("VISU SHAPE",visu.shape-)
        cell1Geometry = cellGeometry[1]
        for cell_1_index in range(tgrid.numberOfCells[1]):

            p1 = cell1_preds[cell_1_index]
            p0 = 1.0 - 0-p1

            u,v = cell_1_bounds[cell_1_index, :]

            if True:#arg3[u]!=arg3[v]:
                coordiantes = cell1Geometry[cell_1_index].__array__()
                #print(coordiantes.shape)
                c0 = int(255.0 * p0 + 0.5)
                c1 = int(255.0 * p1 + 0.5)
                visu[coordiantes[:,0], coordiantes[:,1], :] = c0,c1,0 




















        res_small = seg

        vi_img = 0.0
        ri_img = 0.0
        n_seg =gt_stack.shape[0]

        for i in range(n_seg):

            gt = gt_stack[i,...]
            gt = numpy.require(gt, requirements=['C'])
            res_small = numpy.require(res_small, requirements=['C'])
            vi_img += nifty.ground_truth.VariationOfInformation(gt, res_small).value

            ri_img += nifty.ground_truth.RandError(gt, res_small).index

        vi_img /= n_seg
        ri_img /= n_seg


        beta = 0.5

        # save seg
        filename = os.path.join(self.out_dir,"%s_b%f_seg.h5"%(image_number,beta))
        f = h5py.File(filename, 'w')
        f['data'] = res_small
        f['scores'] = numpy.array([vi_img, ri_img])
        f.close()

        # save visu
        filename = os.path.join(self.out_dir,"%s_b%f_visu.png"%(image_number,beta))
        vigra.impex.writeImage(ol, filename)


        # save visu
        filename = os.path.join(self.out_dir,"%s_overlay.png"%(image_number))
        vigra.impex.writeImage(visu, filename)

        return  vi_img, ri_img





















        # img_small = vigra.sampling.resize(vigra.taggedView(img_raw,'xyc'), seg.shape)
        # ol = nifty.segmentation.segmentOverlay(image=img_small, segmentation=seg, beta=0.24)
        # pylab.imshow(numpy.swapaxes(ol, 0,1))
        # pylab.show()

        # cell1Geometry = cellGeometry[1]
        # for cell_1_index in range(tgrid.numberOfCells[1]):

        #     p1 = cell_1_preds_combined[cell_1_index]
        #     p0 = 1.0 - 0-p1

        #     u,v = cell_1_bounds[cell_1_index, :]

        #     if True:#arg3[u]!=arg3[v]:
        #         coordiantes = cell1Geometry[cell_1_index].__array__()
        #         #print(coordiantes.shape)
        #         c0 = int(255.0 * p0 + 0.5)
        #         c1 = int(255.0 * p1 + 0.5)
        #         visu[coordiantes[:,0], coordiantes[:,1], :] = c0,c1,0 

        # vigra.imshow(visu)
        # vigra.show()




    def predict_bra(self, index):


        res_odict, backward_mapping, cell1_preds, cell0_3_preds, lifted_preds = self.predict_augmented(index)
        cell_1_bounds  = res_odict["cell_1_bounds"]
        sp  = res_odict["sp"]

        res_odict2, backward_mapping2, cell1_preds2, cell0_3_preds2, lifted_preds2 = self.predict_augmented(index, tt_augment=True)
        cell_1_bounds2 = res_odict2["cell_1_bounds"]
        sp2  = res_odict["sp"]

        import nifty.ground_truth
        assert sp.min() == 1
        assert sp2.min() == 1
        overlap = nifty.ground_truth.overlap(segmentation=sp, groundTruth=sp2)

        assert cell1_preds2.min()>=0.0
        assert cell1_preds2.max()<=1.0
        assert cell_1_bounds2.shape[0] == cell1_preds2.shape[0]
        aug_cell1_preds = overlap.transferCutProbabilities(
            cell_1_bounds,
            cell1_preds,
            cell_1_bounds2,
            cell1_preds2
        )
        diff = aug_cell1_preds - cell1_preds
        for cell_1_index in range(aug_cell1_preds.shape[0]):
            if numpy.abs(diff[cell_1_index] > 0.1):
                print(cell1_preds[cell_1_index], aug_cell1_preds[cell_1_index], diff[cell_1_index])

    def run(self, img, preds, tgrid, overseg, cell_1_bounds, cell_1_sizes, image_number, gt_stack):

        # nifty
        import nifty
        import nifty.graph
        import nifty.graph.rag
        import nifty.graph.opt.multicut  as nifty_multicut  
        import numpy
        import nifty

        #preds = preds
        img_big = vigra.sampling.resize(img, [2*s -1 for s in img.shape[0:2]])
        visu = img_big.copy()
        cellGeometry = tgrid.extractCellsGeometry()

        acc_vi_ds = 0.0
        acc_ri_ds = 0.0

        count = 0
        for cell_1_index in range(tgrid.numberOfCells[1]):
            coords = coords = cellGeometry[1][cell_1_index].__array__()
            p1 = preds[cell_1_index]
            p0 = 1.0-p1
            #p1 = res_edges[cell_1_index]
            p0 = 1.0 - p1
            c0 = 255.0*p0
            c1 = 255.0*p1
            visu[coords[:,0], coords[:,1]  ,:] = c0,c1,0
        



        beta  = 0.5

        rag = nifty.graph.rag.gridRag(overseg.astype('uint64'))
        ragw  = numpy.zeros(rag.numberOfEdges)
        ragl = numpy.zeros(rag.numberOfEdges)
        #uv = rag.uvIds()

        ragw[rag.findEdges(cell_1_bounds)] = preds
        ragl[rag.findEdges(cell_1_bounds)] = cell_1_sizes.astype('float32')
        #pos = rag.findEdges(cell_1_bounds)
        p1 = numpy.clip(ragw,0.00001, 0.99999)
        p0 = 1.0 - p1

        length_weight  = ragl
        weights = numpy.log(p0/p1) * (length_weight)  + numpy.log((1.0-beta)/beta) 


        MulticutObjective = rag.__class__.MulticutObjective
        solverFactory = MulticutObjective.multicutIlpCplexFactory()
        objective = MulticutObjective(rag, weights)

        loggingVisitor = MulticutObjective.verboseVisitor(visitNth=1)
        solver = solverFactory.create(objective)
        result = solver.optimize(loggingVisitor)
        res_edges = result[cell_1_bounds[:,0]] != result[cell_1_bounds[:,1]]
        #energies = loggingVisitor.energies()
        #runtimes = loggingVisitor.runtimes()
        # convert graph segmentation

        # to pixel segmentation
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, result)
        overlay = nifty.segmentation.segmentOverlay(img, seg)
        #pylab.imshow(numpy.swapaxes(visu,0,1))
        #pylab.show()


        # resize seg to proper bsd resolution
        res_small = vigra.sampling.resizeImageNoInterpolation(seg.astype('float32'), img.shape[0:2]).astype('uint32')
        res_small = vigra.analysis.labelImage(res_small)


        vi_img = 0.0
        ri_img = 0.0
        n_seg =gt_stack.shape[2]

        for i in range(n_seg):

            gt = gt_stack[...,i]
            gt = numpy.require(gt, requirements=['C'])
            res_small = numpy.require(res_small, requirements=['C'])
            vi_img += nifty.ground_truth.VariationOfInformation(gt, res_small).value

            ri_img += nifty.ground_truth.RandError(gt, res_small).index

        vi_img /= n_seg
        ri_img /= n_seg


        # save seg
        filename = os.path.join(self.out_dir,"%s_b%f_seg.h5"%(image_number,beta))
        f = h5py.File(filename, 'w')
        f['data'] = res_small
        f['scores'] = numpy.array([vi_img, ri_img])
        f.close()

        # save visu
        filename = os.path.join(self.out_dir,"%s_b%f_visu.png"%(image_number,beta))
        vigra.impex.writeImage(overlay, filename)


        # save visu
        filename = os.path.join(self.out_dir,"%s_overlay.png"%(image_number))
        vigra.impex.writeImage(visu, filename)

        return  vi_img, ri_img

        #vigra.segShow(img, seg)
        #vigra.show()