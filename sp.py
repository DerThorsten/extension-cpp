
import nifty 
import vigra
import numpy

# nifty
import nifty.graph.rag      # RAG
import nifty.graph.agglo    # Agglomerative clustering
import nifty.segmentation



def bsd_sp(img_rgb_in, pmap, n_sp=500):
    big_shape = list(pmap.shape)
    shape = [(s+1)//2 for s in big_shape]

    in_shape = list(img_rgb_in.shape[0:2])
    if in_shape != big_shape:
        assert big_shape == [2*s-1 for s in in_shape]
        img_rgb_big = vigra.sampling.resize(vigra.taggedView(img_rgb_in, 'xyc'), big_shape)
    else:
        img_rgb_big = vigra.taggedView(img_rgb_in, 'xyc')
    pmap = vigra.taggedView(pmap, 'xy')

    q = float(255)

    # the pmap
    pmap = pmap.copy()
    pmap_s =  vigra.filters.gaussianSmoothing(pmap,   0.1)
    pmap_m =  vigra.filters.gaussianSmoothing(pmap_s, 1.1)
    pmap_l =  vigra.filters.gaussianSmoothing(pmap_m, 5.1)


    # the gradient
    g = vigra.filters.gaussianGradientMagnitude(img_rgb_big, 0.75).squeeze()
    g -= g.min()
    g /= g.max()


    w = numpy.array([750,3,2,1, 10],dtype='float')
    pmap = pmap* w[0] + pmap_s*w[1] + pmap_m*w[2] + pmap_l*w[3] + g*w[4]
    pmap /= w.sum()


    overseg_uint32, mseg = vigra.analysis.watersheds(pmap.astype('float32'))
    overseg_uint32 -= 1
    overseg_uint64 = numpy.require(overseg_uint32, dtype='uint64', requirements=['C'])


    # vigra.segShow(img_rgb_big, seg.astype('uint32')+1)
    # vigra.show()



    # make the Region adjacency graph (RAG)
    rag = nifty.graph.rag.gridRag(overseg_uint64)
    edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
        rag, pmap, [10,10],1)
    meanEdgeStrength = edge_features[:,0]
    edgeSizes = edge_features[:,1]
    nodeSizes = node_features[:,1]


    # cluster-policy  
    clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
        graph=rag, edgeIndicators=meanEdgeStrength,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes,
        numberOfNodesStop=n_sp, sizeRegularizer=0.2)

    # run agglomerative clustering
    agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(verbose=0, printNth=0)
    nodeSeg = agglomerativeClustering.result()

    # convert graph segmentation
    # to pixel segmentation
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)

    # downsample
    seg =vigra.sampling.resizeImageNoInterpolation(seg.astype('float32'), 
       shape)
    seg = seg.astype('uint32')

    # vigra.segShow(img_rgb_in, seg.astype('uint32')+1)
    # vigra.show()
    # sys.exit()

    return vigra.analysis.labelImage(seg.astype('uint32')), img_rgb_big





if __name__ == "__main__":
    import os


    bsd_root = "/home/tbeier/datasets/BSR/BSDS500/"
    pmap_root = "/home/tbeier/src/holy-edge/hed-data/out"
    split = "train"



    rgb_folder = os.path.join(bsd_root, 'data', 'ima-ges', split)
    pmap_folder = os.path.join(pmap_root, split)



    # get all images filenames

    for file in os.listdir(rgb_folder):
        if file.endswith('.jpg'):
            filename = os.path.join(rgb_folder, file)
            number = file[:-4]

            pmap_filename = os.path.join(pmap_folder, number + '.png')
            img_rgb = vigra.impex.readImage(filename)   
            pmap = vigra.impex.readImage(pmap_filename).squeeze()
            pmap = 1.0 - pmap/255.0
            img_rgb = vigra.sampling.resize(img_rgb, [s*2-1 for s in img_rgb.shape[0:2]])
            bsd_sp(img_rgb, pmap)   
            break
