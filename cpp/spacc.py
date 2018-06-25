import math
from torch import nn
from torch.autograd import Function
import torch

import lltm_cpp
import spacc_cpp

torch.manual_seed(42)


class SpAccAvFunction(Function):
    @staticmethod
    def forward(ctx, input, superpixels):
        max_sp = int(torch.max(superpixels))
        #print("max sp ",max_sp)
        average, labelcount = spacc_cpp.sp_acc_av_forward(input, superpixels, max_sp)

        ctx.save_for_backward(superpixels, labelcount)

        return average

    @staticmethod
    def backward(ctx, grad_out):
        superpixels, labelcount = ctx.saved_variables

        grad_in = spacc_cpp.sp_acc_av_backward(grad_out, superpixels, labelcount)

        return grad_in, None









class SpAccAvModule(nn.Module):

    def __init__(self):
        super(SpAccAvModule, self).__init__()


    def forward(self, input, superpixels):
        recast_to_cuda = False
        cuda_tensors = (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)
        if isinstance(input, cuda_tensors):
            recast_to_cuda = True
            #print("we are on gpu")
            _input = input.cpu()
            _superpixels = superpixels.cpu()
        else:
            _input = input 
            _superpixels = superpixels


        ret =  SpAccAvFunction.apply(_input, _superpixels)
        
        if recast_to_cuda:
            return ret.cuda()
        return ret 


class JunctionPoolFunction(Function):

    def __init__(self):
        super(JunctionPoolFunction, self).__init__()

    @staticmethod
    def forward(ctx, edge_features,  cell_0_bounds):
        #print("forward")
        if isinstance(edge_features, torch.FloatTensor):
            min_max, whereminmax = spacc_cpp.JunctionPoolFloat.forward(
                edge_features, 
                cell_0_bounds
            )
        elif isinstance(edge_features, torch.DoubleTensor):
            min_max, whereminmax = spacc_cpp.JunctionPoolDouble.forward(
                edge_features, 
                cell_0_bounds
            )

        ctx.save_for_backward(whereminmax)

        return min_max

    @staticmethod
    def backward(ctx, grad_out):
        #print("backward")
        whereminmax, = ctx.saved_variables

        if isinstance(grad_out, torch.FloatTensor):
            grad_in, = spacc_cpp.JunctionPoolFloat.backward(grad_out, whereminmax)
        elif isinstance(grad_out, torch.DoubleTensor):
            grad_in, = spacc_cpp.JunctionPoolDouble.backward(grad_out, whereminmax)
        return grad_in, None









        

class JunctionPoolModule(nn.Module):

    def __init__(self):
        super(JunctionPoolModule, self).__init__()


    def forward(self, edge_features, cell_0_bounds):
        assert isinstance(cell_0_bounds,     (torch.IntTensor, torch.cuda.IntTensor))
        recast_to_cuda = False
        cuda_tensors = (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)
        if isinstance(edge_features, cuda_tensors):
            recast_to_cuda = True
            #print("we are on gpu")
            edge_features = edge_features.cpu()
            cell_0_bounds = cell_0_bounds.cpu()
        else:
            pass


        ret =  JunctionPoolFunction.apply(edge_features, cell_0_bounds)
        
        if recast_to_cuda:
            return ret.cuda()
        return ret 







class LabelStatsAccumulatorFunction(Function):

    def __init__(self):
        super(LabelStatsAccumulatorFunction, self).__init__()

    @staticmethod
    def forward(ctx, input, labels, labelcount):
        
        if isinstance(input, torch.FloatTensor):
            statistics, whereminmax = spacc_cpp.LabelStatsAccumulatorFloat.forward(
                input, 
                labels, 
                labelcount
            )
        elif isinstance(input, torch.DoubleTensor):
            statistics, whereminmax = spacc_cpp.LabelStatsAccumulatorDouble.forward(
                input, 
                labels, 
                labelcount
            )

        ctx.save_for_backward(labels, labelcount, whereminmax)

        return statistics

    @staticmethod
    def backward(ctx, grad_out):
        labels, labelcount, whereminmax = ctx.saved_variables

        if isinstance(grad_out, torch.FloatTensor):
            grad_in, = spacc_cpp.LabelStatsAccumulatorFloat.backward(grad_out, labels, labelcount, whereminmax)
        elif isinstance(grad_out, torch.DoubleTensor):
            grad_in, = spacc_cpp.LabelStatsAccumulatorDouble.backward(grad_out, labels, labelcount, whereminmax)
        return grad_in, None, None



class LabelStatsAccumulatorModule(nn.Module):

    def __init__(self):
        super(LabelStatsAccumulatorModule, self).__init__()


    def forward(self, input, labels, labelcount):
        assert isinstance(labels,     (torch.IntTensor, torch.cuda.IntTensor))
        assert isinstance(labelcount, (torch.IntTensor, torch.cuda.IntTensor))
        recast_to_cuda = False
        cuda_tensors = (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)
        if isinstance(input, cuda_tensors):
            recast_to_cuda = True
            #print("we are on gpu")
            input = input.cpu()
            labels = labels.cpu()
            labelcount = labelcount.cpu()
        else:
            pass


        ret =  LabelStatsAccumulatorFunction.apply(input, labels, labelcount)
        
        if recast_to_cuda:
            return ret.cuda()
        return ret 










class Cell12AccModule(nn.Module):


    def __init__(self, in_channels):
        super(Cell12AccModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 3 * in_channels + 1
        self.cell_1_acc = LabelStatsAccumulatorModule()
        self.cell_2_acc = LabelStatsAccumulatorModule()

    def forward(self, input, cell_1_mask, cell_2_mask, cell_1_sizes, cell_2_sizes):
        assert input.size(1) == self.in_channels
        recast_to_cuda = False
        cuda_tensors = (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)
        if isinstance(input, cuda_tensors):
            recast_to_cuda = True
            input = input.cpu()
            cell_1_mask   = cell_1_mask.cpu()
            cell_2_mask   = cell_2_mask.cpu()
            #cell_1_bounds = cell_1_bounds.cpu()
            cell_1_sizes = cell_1_sizes.cpu()
            cell_2_sizes = cell_2_sizes.cpu()
        
        #cell_1_bounds = cell_1_bounds.long()

        cell_1_count = len(cell_1_sizes)
        cell_2_count = len(cell_2_sizes)

        #print("cell_1_acc", input.size(), cell_1_mask.size(), cell_1_sizes().size())
        cell_1_stats =  self.cell_1_acc(input, cell_1_mask, cell_1_sizes)
        #print("cell_2_acc")
        cell_2_stats =  self.cell_2_acc(input, cell_2_mask, cell_2_sizes)
        #print("done")

        # shape |L| x c x  3
        cell_1_stats = cell_1_stats.view(cell_1_count, -1)
        cell_2_stats = cell_2_stats.view(cell_2_count, -1)

        s1 = torch.exp(-1.0*cell_1_sizes.float()) - 0.5
        s2 = torch.exp(-1.0*cell_2_sizes.float()) - 0.5

        cell_1_stats = torch.cat([cell_1_stats, s1.unsqueeze(1)],1)
        cell_2_stats = torch.cat([cell_2_stats, s2.unsqueeze(1)],1)



        assert cell_1_stats.size(1) == self.out_channels
        assert cell_2_stats.size(1) == self.out_channels

        #ret = torch.cat((cell_1_mean, torch.abs(f_u-f_v)), 1)

        if recast_to_cuda:
            return cell_1_stats.cuda(), cell_2_stats.cuda()
        else:
            return cell_1_stats, cell_2_stats


class Cell1AccModule(nn.Module):


    def __init__(self):
        super(Cell1AccModule, self).__init__()

        self.cell_1_acc = LabelStatsAccumulatorModule()
        self.cell_2_acc = LabelStatsAccumulatorModule()

    def forward(self, input, cell_1_mask, cell_2_mask, cell_1_bounds, cell_1_sizes, cell_2_sizes):
        recast_to_cuda = False
        cuda_tensors = (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)
        if isinstance(input, cuda_tensors):
            recast_to_cuda = True
            input = input.cpu()
            cell_1_mask   = cell_1_mask.cpu()
            cell_2_mask   = cell_2_mask.cpu()
            cell_1_bounds = cell_1_bounds.cpu()
            cell_1_sizes = cell_1_sizes.cpu()
            cell_2_sizes = cell_2_sizes.cpu()
        
        cell_1_bounds = cell_1_bounds.long()

        cell_1_count = len(cell_1_sizes)
        cell_2_count = len(cell_2_sizes)

        #print("cell_1_acc", input.size(), cell_1_mask.size(), cell_1_sizes().size())
        cell_1_stats =  self.cell_1_acc(input, cell_1_mask, cell_1_sizes)
        #print("cell_2_acc")
        cell_2_stats =  self.cell_2_acc(input, cell_2_mask, cell_2_sizes)
        #print("done")

        # shape |L| x c x  3
        cell_1_stats = cell_1_stats.view(cell_1_count, -1)
        cell_2_stats = cell_2_stats.view(cell_2_count, -1)

        s1 = torch.exp(-1.0*cell_1_sizes.float()) - 0.5
        s2 = torch.exp(-1.0*cell_2_sizes.float()) - 0.5

        cell_1_stats = torch.cat([cell_1_stats, s1.unsqueeze(1)],1)
        cell_2_stats = torch.cat([cell_2_stats, s2.unsqueeze(1)],1)

        #print(cell_1_mean.detach().numpy())

        u = cell_1_bounds[:,0] - 1
        v = cell_1_bounds[:,1] - 1

        stats_u = cell_2_stats[u, :]
        stats_v = cell_2_stats[v, :]



        #ret = torch.cat((cell_1_mean, torch.abs(f_u-f_v)), 1)

        if recast_to_cuda:
            return cell_1_stats.cuda(), stats_u.cuda(), stats_v.cuda()
        else:
            return cell_1_stats, stats_u, stats_v