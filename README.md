# Kernel-based Localization Project

This repo utilizes standard and quasi-norm kernel-based fusion of heterogeneous data measurements followed by regession method of choice.
It has been shown the use of the quasi-$p$-norm, $p<1$, as a similarity measure, results in improved performance over the standard $p$-norm.
Additionally, the use of multiple kernels, one for each data measurement type --- e.g., time-difference of arrival, received signal strength, and angle of arrival for localization--- further improved performance in comparison to a single kernel.

Repo leverages and builds on [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/)
extensively.  There are two ML model constructs with one focused on single
kernel and the second one focused on multi-kernel.  Creation of kernels
leverages [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/).  If utilize optimal spacing (circle packing) in sampling function, requires pulling files from packomania.com

## Example Usage

Import packages and functions

```
  using Printf, Logging, ProgressMeter, Dates
  using DataFrames, CSV
  using Plots
  pyplot();
  using KernelFunctions, LinearAlgebra

  #core functions - contains MLJ model constructs
  #and kernel functions
  include("kernel_functions.jl")
  #utility sampling functions - if use sphere packing - requires files
  #from packomania.com.
  include("sampling_functions.jl")
```
Read in data and format appropriately

```
  #files for input 
  base_name = "data/20201116_Xpd"  #pickled panda file to load with measurements and associated parameters
  data_file = base_name*".csv"

  #load data
  D_orig = DataFrames.DataFrame(CSV.File(data_file));

  #pull out tdoa, rss and aoa measurements
  X_df=D_orig[:,r"tdoa|rss|aoa"]
  #get aoa columns, convert to radians
  for col=names(X_df[:,r"aoa"]) X_df[!, col]=angle.(X_df[:, col]); end
  #get positions
  y_df=D_orig[:,r"x|y"]
  #y_tb=Tables.rowtable(y_df);

  #recover memory
  D_orig=0;
```


Set up some base regression models 

```
  base_models = [@load RidgeRegressor      pkg ="MLJLinearModels" verbosity=0; #1
                 @load LassoRegressor      pkg ="MLJLinearModels" verbosity=0; #2
                 @load ElasticNetRegressor pkg ="MLJLinearModels" verbosity=0; #3
                 @load KNeighborsRegressor pkg ="ScikitLearn"     verbosity=0; #4
                 @load KNNRegressor verbosity=0];                              #5
  mdl_number = 5;
```
### Single Kernel Regression
Construct, fit, and test Julia ML models.  For single kernel approach, all the
measurements are kernelized using one kernel. In the code below, "y_df" and
"X_df" are dataframes. The former is cartesian location information (x, y) and
the latter has a set of measurements associated with each location.  Data is
kernelized and then regressed to estimate location.

```
  t_start = now()
  #declare composite model including core model and kernel
  mlj_enr_x = SingleKernelRegressor(mdl= base_models[mdl_number]())
  mlj_enr_y = SingleKernelRegressor(mdl= base_models[mdl_number]());

  #set model λ_kern
  mlj_enr_x.λ_kern=mlj_enr_y.λ_kern=999.0  #999.0
  #set model p
  mlj_enr_x.p=mlj_enr_y.p=0.20

  #set lambda/gamma... can comment out and use default settings
  #set some default settings for various models
  if mdl_number == 1 || mdl_number == 2
      mlj_enr_x.mdl.lambda = 1e-1
      mlj_enr_y.mdl.lambda = 1e-1
  elseif mdl_number == 3
      mlj_enr_x.mdl.lambda = mlj_enr_y.mdl.lambda = 1e-1
      mlj_enr_x.mdl.gamma  = mlj_enr_y.mdl.gamma  = 1e-1
  elseif mdl_number == 4
      mlj_enr_x.mdl.n_neighbors = mlj_enr_y.mdl.n_neighbors = 1           #default is 5
      mlj_enr_x.mdl.n_jobs = mlj_enr_y.mdl.n_jobs   = 6                   #default is 1
      mlj_enr_x.mdl.leaf_size = mlj_enr_y.mdl.leaf_size = 10              #default is 30
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = "kd_tree"       #default is auto
  elseif mdl_number ==5
      mlj_enr_x.mdl.K         = mlj_enr_y.mdl.K         = 1
      mlj_enr_x.mdl.leafsize  = mlj_enr_y.mdl.leafsize  = 10
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = :kdtree
  else
      println("check model choice")
  end

  #re-split into test, train indices... getting indices for each row
  #train_idx, valid_idx, test_idx = partition(eachindex(y_df[:,1]), 0.1, 0.1, shuffle=true, rng=1235); #70:20:10 split, want 30% for testing
  train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true, rng=1235); #70:20:10 split

  #create machines
  opt_mc_x = machine(mlj_enr_x, X_df[train_idx,:], y_df[train_idx, :x])
  opt_mc_y = machine(mlj_enr_y, X_df[train_idx,:], y_df[train_idx, :y])

  #fit the machines
  MLJ.fit!(opt_mc_x, verbosity=1)
  MLJ.fit!(opt_mc_y, verbosity=1);

  #predict
  yhat_x = MLJ.predict(opt_mc_x, X_df[test_idx,:])
  yhat_y = MLJ.predict(opt_mc_y, X_df[test_idx,:]);

  #mean euclidean distance error
  current_mee   = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_std   = std(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_mee_x = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2)))
  current_mee_y = mean(sqrt.((yhat_y- y_df[test_idx, :y]).^(2)))

  #see results
  @printf("The avg error (%0.2fm) and std (%0.2fm) using model: %s against %s data\n", current_mee, current_std, MLJModelInterface.name(mlj_enr_x.mdl), data_file)
  @printf("Avg error for x and y dimensions: %5.2f | %5.2f\n", current_mee_x, current_mee_y)
  @printf("%s\n",(now()-t_start))
  pprint(mlj_enr_x.mdl)
```

### Multi-Kernel Regression

Construct, fit, and test Julia ML models.  For multi-kernel approach, each set of 
measurements are kernelized independently. In the code below, "y_df" and
"X_df" are dataframes. The former is cartesian location information (x, y) and
the latter has a set of measurements {TDoA, RSS, AoA} associated with each location.  Data is
kernelized and then regressed to estimate location.

```
  t_start=now()

  #declare composite model including core model and kernel
  mlj_enr_x = MultipleKernelRegressor(mdl= base_models[mdl_number]())
  mlj_enr_y = MultipleKernelRegressor(mdl= base_models[mdl_number]());

  #set model p, λ_kern (use 999.0 flag), and feature count
  p        = [0.385 0.385]
  λ_kern   = [999.0 999.0]
  f_counts = [112 24]
  mlj_enr_x.p        = mlj_enr_y.p        = p
  mlj_enr_x.λ_kern   = mlj_enr_y.λ_kern   = λ_kern
  mlj_enr_x.f_counts = mlj_enr_y.f_counts = f_counts

  #set lambda/gamma... can comment out and use default settings
  #set some default settings for various models
  if mdl_number == 1 || mdl_number == 2
      mlj_enr_x.mdl.lambda = 1e-1
      mlj_enr_y.mdl.lambda = 1e-1
  elseif mdl_number == 3
      mlj_enr_x.mdl.lambda = mlj_enr_y.mdl.lambda = 1e-1
      mlj_enr_x.mdl.gamma  = mlj_enr_y.mdl.gamma  = 1e-1
  elseif mdl_number == 4
      mlj_enr_x.mdl.n_neighbors = mlj_enr_y.mdl.n_neighbors = 1           #default is 5
      mlj_enr_x.mdl.n_jobs    = mlj_enr_y.mdl.n_jobs    = 6               #default is 1
      mlj_enr_x.mdl.leaf_size = mlj_enr_y.mdl.leaf_size = 10              #default is 30
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = "kd_tree"       #default is auto
  elseif mdl_number == 5
      mlj_enr_x.mdl.K         = mlj_enr_y.mdl.K           = 1
      mlj_enr_x.mdl.leafsize  = mlj_enr_y.mdl.leafsize    = 5
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm   = :kdtree
  else
      println("check model choice")
  end

  #re-split into test, train indices... getting indices for each row
  #train_idx, valid_idx, test_idx = partition(eachindex(y_df[:,1]), 0.6, 0.1, shuffle=true, rng=1235); #70:20:10 split, want 30% for testing
  train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true)#, rng=1235); #70:20:10 split

  #create machines
  opt_mc_x = machine(mlj_enr_x, X_df[train_idx,:], y_df[train_idx, :x])
  opt_mc_y = machine(mlj_enr_y, X_df[train_idx,:], y_df[train_idx, :y])

  #fit the machines
  MLJ.fit!(opt_mc_x, verbosity=1)
  MLJ.fit!(opt_mc_y, verbosity=1);

  #predict
  yhat_x = MLJ.predict(opt_mc_x, X_df[test_idx,:])
  yhat_y = MLJ.predict(opt_mc_y, X_df[test_idx,:]);

  #mean euclidean distance error
  current_mee   = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_std   = std(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_mee_x = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2)))
  current_mee_y = mean(sqrt.((yhat_y- y_df[test_idx, :y]).^(2)))

  #see results
  @printf("The avg error (%0.2fm) and std (%0.2fm) using model: %s against %s data\n", current_mee, current_std, MLJModelInterface.name(mlj_enr_x.mdl), data_file)
  @printf("Avg error for x and y dimensions: %5.2f | %5.2f\n", current_mee_x, current_mee_y)
  @printf("%s\n",(now()-t_start))
  pprint(mlj_enr_x.mdl)
```

### Optimal Training Spacing

To selectively chose a subset of measurement locations for training, there is a set of utility
functions to periodically choose from given measurements.  One approach is
sample base on given order of measurements.  Another approach via circle packing 
is to choose set number of measurements from locations with maximal
spacing.  Note that function assumes rectangular area when passing in
measurement locations.

First example is periodic sampling based on order of measurements:

```
  #periodic sampling based on order of sample locations
  train_idx = periodic_sampling(train_percentage, size(y_df)[1])
  _, test_idx = partition(setdiff(eachindex(y_df[:,1]), train_idx), 0.7, shuffle=true); #70:20:10 split want 30% for test, remove train_idx index

```

Second example is periodic sample based on optimal spacing.  First load
packing coordinates from set of files.  This is a one time operation.

```
  #loading of packing coordinates
  packing_df = DataFrames.DataFrame()
  file_location_packing = "./data/packing/"
  #set of possible numbers that will be iterated/tested
  for train_percentage in  0.0025.*2.0.^(range(0,5,step=1))
      train_idx = periodic_sampling(train_percentage, size(y_df)[1])
      print(size(train_idx,1), " ")#," ", train_idx)
      
      idx = size(train_idx,1)
      #find file with number of circles tha match number of training points
      filename = filter!(s->occursin(r"crc"*string(idx)*"_0.6",s),readdir(file_location_packing))
      print(filename," ")
      temp_df = DataFrames.DataFrame(CSV.File(file_location_packing*filename[1], header=["grp", "x", "y"], delim=' ', ignorerepeated=true))
      temp_df[!, :grp]=temp_df[!, :grp].*0 .+ idx
      append!(packing_df, temp_df)
  end
  packing_df = DataFrames.groupby(packing_df, :grp);
```
Now that coordinates are loaded, choose optimal spacing of measurement locations:

```
  #sampling based on optimized spacing (packing)
  train_idx = periodic_sampling(train_percentage, size(y_df)[1], y_df, packing_df, rand_float=1.0)
  _, test_idx = partition(setdiff(eachindex(y_df[:,1]), train_idx), 0.7, shuffle=true); #70:20:10 split want 30% for test, remove train_idx index

```
