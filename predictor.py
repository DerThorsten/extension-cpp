import torch
import vigra
import os
import h5py
import numpy

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)

    return x[tuple(indices)]

class Predictor(object):
    def __init__(self, model, ds, output_folder):
        self.model = model.eval()
        self.ds = ds
        self.out_dir = os.path.join(output_folder, self.ds.split)
        try:
            os.stat(self.out_dir)
        except:
            os.mkdir(self.out_dir)    



    def predict(self, index):
        res_odict = self.ds.getitemimpl(index=index, return_test_data=True)




        tgrid        = res_odict["tgrid"]
        img_raw      = res_odict["img_raw"]
        sp           = res_odict["sp"]
        image_number = res_odict["image_number"]
        gt_stack     = res_odict["gt_stack"]

        #############################################
        # GET X / INPUT TENSORS
        # AND PAD THEM
        #############################################
        tlist = []
        for i,key in enumerate(res_odict.keys()):
            tlist.append(res_odict[key][None,...])
            if i + 1 == self.ds.num_inputs():
                break
        ttuple = tuple(tlist)

        #############################################
        # DO THE PREDICTION
        #############################################
        cell1_preds, cell0_3_preds, cell0_4_preds = self.model(
            *ttuple[0:self.ds.num_inputs()]
        )

        #############################################
        # DETACH RESULTS
        #############################################
        cell1_preds   = cell1_preds.detach().numpy()
        cell0_3_preds = cell0_3_preds.detach().numpy()
        cell0_4_preds = cell0_4_preds.detach().numpy()
        
        #############################################
        # DO STUFF WITH RESULTS!
        #############################################
        for i in range(cell0_3_preds.shape[0]):
            print("C03_%d"%i, cell0_3_preds[i,...])

        for i in range(cell0_3_preds.shape[0]):
            print("C03_%d"%i, numpy.round(cell0_3_preds[i,...]))

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



        cell_1_preds_wsum = numpy.zeros(tgrid.numberOfCells[1])
        cell_1_preds_w    = numpy.zeros(tgrid.numberOfCells[1])

        # from cell 1
        for cell_1_index in range(tgrid.numberOfCells[1]):
            p1 = cell1_preds[cell_1_index]
            cell_1_preds_wsum[cell_1_index] += 1.0*p1
            cell_1_preds_w[cell_1_index] += 1.0

        # from cell 0
        i3 = 0
        i4 = 0
        for cell_0_index in range(tgrid.numberOfCells[0]):
            bounds = cell_0_bounds[cell_0_index,:]
            jsize = 4
            if bounds[3] == 0:
                jsize = 3
                bounds = bounds[0:3]
                cell_1_pred = cell0_3_preds[i3,...]
                i3 += 1
            else:
                cell_1_pred = cell0_4_preds[i4,...]
                i4 += 1

            cell_1_indices = bounds  - 1

            for cell_1_index,p1 in zip(cell_1_indices, cell_1_pred):
                
                if jsize == 3:
                    cell_1_preds_wsum[cell_1_index] += 0.3*p1
                    cell_1_preds_w[cell_1_index] += 0.3
                else:
                    cell_1_preds_wsum[cell_1_index] += 0.1*p1
                    cell_1_preds_w[cell_1_index] += 0.1

        cell_1_preds_combined = cell_1_preds_wsum / cell_1_preds_w

        visu = img_raw_big.copy()












        cell1Geometry = cellGeometry[1]
        for cell_1_index in range(tgrid.numberOfCells[1]):

            p1 = cell_1_preds_combined[cell_1_index]
            p0 = 1.0 - p1

            coordiantes = cell1Geometry[cell_1_index].__array__()
            print(coordiantes.shape)
            c0 = int(255.0 * p0 + 0.5)
            c1 = int(255.0 * p1 + 0.5)
            visu[coordiantes[:,0], coordiantes[:,1], :] = c0,c1,0 

        vigra.imshow(visu)
        vigra.show()


    

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