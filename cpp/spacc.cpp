#include <torch/torch.h>
#include <iostream>
#include <vector>

#define IMG_2D_CHANNEL_AXIS 1


template<class float_t>
struct JunctionPool
{
    typedef float_t float_type;

   
    const static std::size_t min_i = 0;
    const static std::size_t max_i = 1;

    static 
    std::vector<at::Tensor> forward
    (
        at::Tensor edge_features,
        at::Tensor cell_0_bounds
    ) 
    {
        const auto n_edges = edge_features.size(0);
        const auto n_channels =  edge_features.size(1);

        at::Tensor result,min_max;
        if(std::is_same<float, float_type>::value){
            min_max = at::zeros(torch::CPU(at::kFloat), 
            {
                n_edges,
                n_channels,
                2
            });
        }
        if(std::is_same<double, float_type>::value){
            min_max = at::zeros(torch::CPU(at::kDouble), 
            {
                n_edges,
                n_channels,
                2
            });
        }


        auto where_min_max = at::zeros(torch::CPU(at::kInt), 
        {
            n_edges,
            n_channels,
            2
        });




        
        auto edge_features_accessor = edge_features.accessor<float_type, 2>();
        auto cell_0_bounds_accessor = cell_0_bounds.accessor<int, 2>();
      
        auto min_max_accessor = min_max.accessor<float_type, 3>();
        auto where_min_max_accessor  = where_min_max.accessor<int, 3>();

        
        // normalize
        for(int edge_index = 0; edge_index < n_edges; ++edge_index)
        {
            

            // walk over channels
            for(int c = 0; c < n_channels; c++) 
            {
      
                min_max_accessor[edge_index][c][min_i] = edge_features_accessor[edge_index][c];
                min_max_accessor[edge_index][c][max_i] = edge_features_accessor[edge_index][c];
                where_min_max_accessor[edge_index][c][min_i] = edge_index; 
                where_min_max_accessor[edge_index][c][max_i] = edge_index;
            }
        } 
        // loop over all junctions
        for(int junction_index=0; junction_index<cell_0_bounds.size(0); ++ junction_index)
        {
            const auto n_b = (cell_0_bounds_accessor[junction_index][3] == 0) ? 3 : 4;

            // walk over channels
            for(int c = 0; c < n_channels; c++) 
            {
                float min_j =  std::numeric_limits<float_type>::infinity();
                int   min_b = -1;

                float max_j = -std::numeric_limits<float_type>::infinity();
                int   max_b = -1;

                for(int b=0; b<n_b; ++b)
                {
                    auto edge_index = cell_0_bounds_accessor[junction_index][b] - 1;

                    const auto val = edge_features_accessor[edge_index][c];
                    if(val < min_j)
                    {
                        min_b = b;
                        min_j = val;
                    }
                    if(val > max_j)
                    {
                        max_b = b;
                        max_j = val;
                    }
                }

                for(int b=0; b<n_b; ++b)
                {
                    auto edge_index = cell_0_bounds_accessor[junction_index][b] - 1;
                    if(min_j < min_max_accessor[edge_index][c][min_i] )
                    {
                        min_max_accessor[edge_index][c][min_i] = min_j;
                        where_min_max_accessor[edge_index][c][min_i] = 
                            cell_0_bounds_accessor[junction_index][min_b] - 1;
                    }
                    if(max_j > min_max_accessor[edge_index][c][max_i] )
                    {
                        min_max_accessor[edge_index][c][max_i] = max_j;
                        where_min_max_accessor[edge_index][c][max_i] = 
                            cell_0_bounds_accessor[junction_index][max_b] - 1;
                    }
                }
            }
        }

        return {
            min_max,
            where_min_max
        };
    }

    static 
    std::vector<at::Tensor> backward
    (
        at::Tensor grad_out,
        at::Tensor where_min_max
    ) 
    {
        const auto n_edges = grad_out.size(0);
        const auto n_channels =  grad_out.size(1);

        at::Tensor grad_in;
        if(std::is_same<float, float_type>::value){
            grad_in = at::zeros(torch::CPU(at::kFloat), 
            {
                n_edges,
                n_channels
            });
        }
        if(std::is_same<double, float_type>::value){
            grad_in = at::zeros(torch::CPU(at::kDouble), 
            {
                n_edges,
                n_channels,
            });
        }





        
        auto grad_in_accessor = grad_in.accessor<float_type, 2>();
        auto grad_out_accessor = grad_out.accessor<float_type, 3>();
        auto where_min_max_accessor  = where_min_max.accessor<int, 3>();

        
        // normalize
        for(int edge_index = 0; edge_index < n_edges; ++edge_index)
        {
            

            // walk over channels
            for(int c = 0; c < n_channels; c++) 
            {
                auto b_min = where_min_max_accessor[edge_index][c][min_i]; 
                auto b_max = where_min_max_accessor[edge_index][c][max_i];
                auto gout_min = grad_out_accessor[edge_index][c][min_i];
                auto gout_max = grad_out_accessor[edge_index][c][max_i];
                grad_in_accessor[b_min][c] += gout_min;
                grad_in_accessor[b_max][c] += gout_max;
            }
        } 
        
        return {
            grad_in
        };
    }

};




template<class float_t>
struct LabelStatsAccumulator
{
    typedef float_t float_type;

    const static std::size_t mean_i = 0;
    const static std::size_t min_i = 1;
    const static std::size_t max_i = 2;

    static 
    std::vector<at::Tensor> forward
    (
        at::Tensor input,
        at::Tensor superpixels,
        at::Tensor labelcount
    ) 
    {
        const auto max_label = labelcount.size(0);
        const auto n_channels = input.size(IMG_2D_CHANNEL_AXIS);

        at::Tensor statistics;
        if(std::is_same<float, float_type>::value){
            statistics = at::zeros(torch::CPU(at::kFloat), 
            {
                max_label,
                n_channels,
                3
            });
        }
        if(std::is_same<double, float_type>::value){
            statistics = at::zeros(torch::CPU(at::kDouble), 
            {
                max_label,
                n_channels,
                3
            });
        }


        auto where_min_max = at::zeros(torch::CPU(at::kInt), 
        {
            max_label,
            n_channels,
            4
        });

        auto labelcount_accessor  = labelcount.accessor<int, 1>();
        auto superpixels_accessor = superpixels.accessor<int, 2>();
        auto input_accessor =       input.accessor<float_type, 4>();
      

        auto statistics_accessor  = statistics.accessor<float_type, 3>();
        auto where_min_max_accessor  = where_min_max.accessor<int, 3>();
        // initialize
        // normalize
        for(int label = 1; label <= max_label; ++label)
        {
            
            const uint  label_index = label - 1;

            // walk over channels
            for(int c = 0; c < n_channels; c++) 
            {
                // average
                /*is fine*///statistics_accessor[label_index][c][mean_i] = 0;

               //  // min
               statistics_accessor[label_index][c][min_i] =  std::numeric_limits<float>::infinity();
               //  // max
               statistics_accessor[label_index][c][max_i] = -std::numeric_limits<float>::infinity();
            }
        
        } 

        // count for each label
        for(int x0 = 0; x0 < superpixels.size(0); x0++)
        for(int x1 = 0; x1 < superpixels.size(1); x1++) 
        {
            const auto label = superpixels_accessor[x0][x1];
            if(label != 0)
            {
                const uint  label_index = label - 1;

                //++labelcount_accessor[label_index];

                // walk over channels
                for(int c = 0; c < n_channels; c++) 
                {
                    const auto val = input_accessor[0][c][x0][x1];

    
                    auto old_min = statistics_accessor[label_index][c][min_i];
                    auto old_max = statistics_accessor[label_index][c][max_i];
                
                    if(val < old_min)
                    {
                        where_min_max_accessor[label_index][c][0] = x0;
                        where_min_max_accessor[label_index][c][1] = x1;
                        // min
                        statistics_accessor[label_index][c][min_i] = val;
                    }
                    if(val > old_max)
                    {
                        where_min_max_accessor[label_index][c][2] = x0;
                        where_min_max_accessor[label_index][c][3] = x1;
                        // max
                        statistics_accessor[label_index][c][max_i] = val;
                    }
                    // average
                    statistics_accessor[label_index][c][mean_i] += val;
                    
                }
            }
        }

        // normalize
        for(int label = 1; label <= max_label; ++label)
        {
            
            const uint  label_index = label - 1;

            const auto count = labelcount_accessor[label_index];

            // walk over channels
            for(int c = 0; c < n_channels; c++) 
            {
                statistics_accessor[label_index][c][mean_i] /= count;
            }
        
        } 
        return 
        {
            statistics,
            where_min_max
        };
    }

    static 
    std::vector<at::Tensor> backward
    (
        at::Tensor grad_out,
        at::Tensor superpixels,
        at::Tensor labelcount,
        at::Tensor where_min_max
    ) 
    {

        const auto max_label = labelcount.size(0);

        const auto n_channels = grad_out.size(1);

        // the statistics average,min,max
        at::Tensor grad_in;

        if(std::is_same<float, float_type>::value){


            grad_in = at::zeros(torch::CPU(at::kFloat), 
            {
                1,
                n_channels,
                superpixels.size(0),
                superpixels.size(1)
            });
        }
        else{
            grad_in = at::zeros(torch::CPU(at::kDouble), 
            {
                1,
                n_channels,
                superpixels.size(0),
                superpixels.size(1)
            });
        }


  
        auto grad_out_accessor    = grad_out.accessor<float_type, 3>();
        auto grad_in_accessor     = grad_in.accessor<float_type, 4>();
        auto labelcount_accessor  = labelcount.accessor<int, 1>();
        auto superpixels_accessor = superpixels.accessor<int, 2>();
        auto where_min_max_accessor  = where_min_max.accessor<int, 3>();

      
        

        
        for(int label = 1; label <= max_label; ++label)
        {
            const auto label_index = label -1 ;

            for(int c = 0; c < n_channels; c++)
            {

                const auto min_x0 = where_min_max_accessor[label_index][c][0];
                const auto min_x1 = where_min_max_accessor[label_index][c][1];
                const auto max_x0 = where_min_max_accessor[label_index][c][2];
                const auto max_x1 = where_min_max_accessor[label_index][c][3];

                grad_in_accessor[0][c][min_x0][min_x1] += grad_out_accessor[label_index][c][min_i];
                grad_in_accessor[0][c][max_x0][max_x1] += grad_out_accessor[label_index][c][max_i];
            }
        } 

        
        for(int x0 = 0; x0 < superpixels.size(0); x0++)
        for(int x1 = 0; x1 < superpixels.size(1); x1++) 
        {
            const auto label = superpixels_accessor[x0][x1];
            if(label != 0)
            {
                const uint  label_index = label - 1;

                const auto count = labelcount_accessor[label_index];

                // walk over channels
                for(int c = 0; c < n_channels; c++) 
                {
                    grad_in_accessor[0][c][x0][x1] += (grad_out_accessor[label_index][c][mean_i] / count);
                }
            }
        }

        return {
            grad_in
        };  
    }
};







py::class_<JunctionPool<float>>(m, "JunctionPoolFloat")
    .def(py::init<>())
        .def_static("forward", &JunctionPool<float>::forward)
        .def_static("backward", &JunctionPool<float>::backward)
    ;
py::class_<JunctionPool<double>>(m, "JunctionPoolDouble")
    .def(py::init<>())
        .def_static("forward", &JunctionPool<double>::forward)
        .def_static("backward", &JunctionPool<double>::backward)
    ;



py::class_<LabelStatsAccumulator<float>>(m, "LabelStatsAccumulatorFloat")
    .def(py::init<>())
        .def_static("forward", &LabelStatsAccumulator<float>::forward)
        .def_static("backward", &LabelStatsAccumulator<float>::backward)
    ;
py::class_<LabelStatsAccumulator<double>>(m, "LabelStatsAccumulatorDouble")
    .def(py::init<>())
        .def_static("forward", &LabelStatsAccumulator<double>::forward)
        .def_static("backward", &LabelStatsAccumulator<double>::backward)
    ;

}
