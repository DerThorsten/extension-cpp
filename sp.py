
import nifty 
import vigra
import numpy

# nifty
import nifty.graph.rag      # RAG
import nifty.graph.agglo    # Agglomerative clustering
import nifty.segmentation
import functools


def sanitize(array, dtype=None):
    array = array.view(numpy.ndarray)
    if dtype is None:
        array = numpy.require(array, requirements=['C'])
    else:
        array = numpy.require(array, requirements=['C'], dtype=dtype)
    return array






def rval(n, range, train, sigma):
    if not train:
        return n
    else:
        val = numpy.random.normal(loc=n, scale=sigma)
        val = numpy.clip(val, range[0], range[1])

    return val


def bsd_sp(img_rgb_in, pmap, n_sp=500, train=False, tt_augment=False):


    augrand = functools.partial(rval, train=(train or tt_augment))


    #print("INSHAPE", img_rgb_in.shape)
    img_rgb_in = numpy.rollaxis(img_rgb_in,0,3)
    #print("afer", img_rgb_in.shape)
    #print("pm", pmap.shape)
    assert img_rgb_in.shape[2] == 3

    big_shape = list(pmap.shape)
    shape = [(s+1)//2 for s in big_shape]

    in_shape = list(img_rgb_in.shape[0:2])
    if in_shape != big_shape:
        assert big_shape == [2*s-1 for s in in_shape]
        img_rgb_big = vigra.sampling.resize(vigra.taggedView(img_rgb_in, 'xyc'), big_shape)
    else:
        img_rgb_big = vigra.taggedView(img_rgb_in, 'xyc')
    #pmap = vigra.taggedView(pmap, 'xy')

    q = float(255)

    #pmap = vigra.sampling.resize(pmap, [2*s - 1 for s in pmap.shape])
    #img_rgb_big = vigra.sampling.resize(img_rgb_big, [2*s - 1 for s in img_rgb_big.shape[0:2]])

    # the pmap
    pmap =    sanitize(pmap.copy())
    pmap_s =  sanitize(vigra.filters.gaussianSmoothing(pmap,   augrand( 1.0*0.1, sigma=0.2, range=[0.05, 10.7])  ))
    pmap_m =  sanitize(vigra.filters.gaussianSmoothing(pmap_s, augrand( 1.0*1.1, sigma=0.2, range=[0.05, 10.7])  ))
    pmap_l =  sanitize(vigra.filters.gaussianSmoothing(pmap_m, augrand( 1.0*5.1, sigma=0.2, range=[0.05, 10.7])  ))


    # the gradient
    g = vigra.filters.gaussianGradientMagnitude(img_rgb_big, 3.2).squeeze()
    g = vigra.filters.gaussianSmoothing(g, 0.2)
    g = sanitize(g)
    g -= g.min()
    g /= g.max()
    
    w = numpy.array(
    [
        augrand(3.0,   sigma=2.1, range=[0, 10000]),
        augrand(0.3,   sigma=0.7,  range=[0, 3]),
        augrand(2.0,   sigma=1.1,  range=[0, 4]),
        augrand(1.0,   sigma=0.3,  range=[0, 3]),
        augrand(80.0,  sigma=20.1,  range=[0, 120]),
    ],dtype='float')
    #w[:] = 0
    #w[-1] = 1.0 
    ws_map = pmap* w[0] + pmap_s*w[1] + pmap_m*w[2] + pmap_l*w[3] + g*w[4]
    ws_map /= w.sum()


    overseg_uint32, mseg = vigra.analysis.watersheds(ws_map.astype('float32'))
    overseg_uint32 -= 1
    overseg_uint64 = sanitize(overseg_uint32, dtype='uint64')
    


    # vigra.segShow(img_rgb_big, overseg_uint32.astype('uint32')+1)
    # vigra.show()



    w = numpy.array(
    [
        augrand(3.0,   sigma=1.1,  range=[0, 10000]),
        augrand(1.3,   sigma=1.1,  range=[0, 3]),
        augrand(1.0,   sigma=1.1,  range=[0, 4]),
        augrand(1.0,   sigma=1.1,  range=[0, 3]),
        augrand(0.1,   sigma=0.3,  range=[0, 3]),
    ],dtype='float')
    w[:] = 0
    w[0] = 1.0
    w[-1] = 1.0 
    pacc = pmap* w[0] + pmap_s*w[1] + pmap_m*w[2] + pmap_l*w[3] + g*w[4]
    pacc /= w.sum()



    # make the Region adjacency graph (RAG)
    rag = nifty.graph.rag.gridRag(overseg_uint64)
    edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
        rag, pacc, [10,10],1)
    meanEdgeStrength = edge_features[:,0]
    edgeSizes = edge_features[:,1]
    nodeSizes = node_features[:,1]


    sizeRegularizer = augrand(0.85, sigma=0.02, range=[0.20, 1.0])
    #print("sizeRegularizer",sizeRegularizer)
    #sizeRegularizer = 0.25
    if tt_augment:
        mx = min(rag.numberOfNodes-1, 2000)
        numberOfNodesStop = int(augrand(n_sp, sigma=200.0, range=[200, mx]))
        #numberOfNodesStop = numpy.random.randint(10, 2500)
        #numberOfNodesStop = min(numberOfNodesStop, rag.numberOfNodes-1)
    else:
        mx = min(rag.numberOfNodes-1, 1800)
        numberOfNodesStop = int(augrand(n_sp, sigma=200.0, range=[500,  mx]))
    #("numberOfNodesStop",numberOfNodesStop)
    #print("sizeRegularizer",sizeRegularizer)
    #print("numberOfNodesStop",numberOfNodesStop)
    # cluster-policy  
    #numberOfNodesStop = 100
    clusterPolicy = nifty.graph.agglo.edgeWeightedClusterPolicy(
        graph=rag, edgeIndicators=meanEdgeStrength,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes,
        numberOfNodesStop=numberOfNodesStop, sizeRegularizer=sizeRegularizer)









    # run agglomerative clustering
    agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(verbose=0, printNth=0)
    nodeSeg = agglomerativeClustering.result()

    # convert graph segmentation
    # to pixel segmentation
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)



    # vigra.segShow(img_rgb_big, seg.astype('uint32')+1)
    # vigra.show()



    if False:
        # and again
        # make the Region adjacency graph (RAG)
        rag = nifty.graph.rag.gridRag(seg)
        edge_features, node_features = nifty.graph.rag.accumulateMeanAndLength(
            rag, pmap, [10,10],1)
        meanEdgeStrength = edge_features[:,0]
        edgeSizes = edge_features[:,1]
        nodeSizes = node_features[:,1]


        minimumNodeSize = int(augrand(5*5, sigma=1.0, range=[3*3, 15*15]))
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








    
    # downsample
    seg =vigra.sampling.resizeImageNoInterpolation(seg.astype('float32'), 
       shape)
    seg = seg.astype('uint32')

    #print("SEG", )
    #vigra.segShow(vigra.sampling.resize(vigra.taggedView(img_rgb_big,'xyc'), seg.shape), vigra.taggedView(seg,'xy'))
    #vigra.show()

    return vigra.analysis.labelImage(seg.astype('uint32'))





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
