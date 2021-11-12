#using <add in any packages as necessary>


###############################################################################
#  Sampling functions used to selectively pick different training locations. 
#   - method to pick based on order of measurements
#   - method to pick based on optimal sphere packing
#  
#  Packing parameter function to load coordinates based on files from
#  packomania.com
###############################################################################

"""
  load\\_packing\\_coords(file\\_location\\_str, num\\_locs\\_IntVec, ratio\\_float)

  Looks at files in <file\\_locations\\_st> and finds ones that corresponds to
  the passed integers in <num\\_locs>.  Here, assumes that <ratio> is 0.6. 
  Review https://packomania.com as standard ratios {0.1 : 0.1 : 1.0}.

  file\\_location\\_str is string with file location, e.g., "./data/packing/".
  Corresponds to location of files with sphere packing coordinates.

  num\\_locs\\_IntVec is vector of integers, e.g., [9, 18, 37]. Corresponds to the
  number of spheres that can be packed in rectangular with width:height ratio
  of <ratio\\_float>.  Together with <ratio\\_float>, determines which info is loaded.

  ratio\\_float is a float that corresponds to ratio (width:height) of rectangular 
  that encompasses all possible location values

  Returns as grouped dataframe with each group (reflected by group label) 
  corresponding to an integer in the provide <num\\_locs\\_IntVec>
"""
function load_packing_coords(file_location::String, num_locs::Vector{Int64}; ratio::Float64=0.6)
  
  #create empty dataframe to return
  packing_df = DataFrames.DataFrame()
  
  #cycle over each integer
  for idx in num_locs
    #find and get filename for desired packing coordinates
    filename = filter!(s->occursin(r"crc"*string(idx)*"_"*string(ratio),s),readdir(file_location_packing))
    if size(filename,1) == 0 error("no files matched parameters, num pts="*string(idx)*", ratio="*string(ratio)) end
    @debug idx print(filename," ")
    temp_df = DataFrames.DataFrame(CSV.File(file_location_packing*filename[1], header=["grp", "x", "y"], delim=' ', ignorerepeated=true))
    temp_df[!, :grp]=temp_df[!, :grp].*0 .+ idx
    append!(packing_df, temp_df)
  end
  
  #create grouped dataframe
  packing_df = DataFrames.groupby(packing_df, :grp)

  return packing_df
end

"""
    periodic\\_sampling(s\\_int, t\\_int)

    Periodically sample the range(1, t\\_int, step=1) of values and return. Used
    for generating a periodically sampled index.

    s\\_int is number of values to sample out of total t\\_int values
"""
function periodic_sampling(num_samples::Integer, num_total::Integer)

    sample_freq = num_total/num_samples

    return round.(Int, sample_freq.*collect(range(1, num_samples, step=1)))
end

"""
    periodic\\_sampling(s\\_float, t\\_int)

    Periodically sample the range(1, t\\_int, step=1) of values and return. Used
    for generating a periodically sampled index.

    s\\_float is percentage of total t\\_int values to sample
"""
function periodic_sampling(percentage_samples::Float64, num_total::Integer)

    sample_points = round(Int, percentage_samples*num_total)
    sample_freq   = 1.0/percentage_samples

    return floor.(Int, sample_freq.*collect(range(0, sample_points-1, step=1))).+1
end

"""
    periodic\\_sampling(s\\_float, t\\_int, locs\\_dataframe, pack\\_coords\\_dataframe, rand\\_float)

    Periodically sample the available locations based on max packing concept, i.e.,
    coordinates are chosen that maximums radius between sampled locations.  Assumes
    that locs\\_dataframe is all the possible sampled locations within a rectangular
    area.  pack\\_coords are the coordinates for maximum packing given in normalized
    units, see http://packomania.com/.

    s\\_float is percentage of total t\\_int values to sample.

    locs is assumed to be Nx2 dataframe with header names [:x, :y]

    pack\\_coords is assumped to be a grouped dataframe.  Each group is based on
    number of coords, e.g., groups 9, 18, 35, 71.

    rand\\_float (default=0.0) sets the maximum uniformly distributed random offset. This parameter
    is used to address possibility that chosen packing location is a particularly
    bad set of measurements.

    group\\_flag (default=false) grabs all values within rand\\_float dimensional offset
    of a given packing location.  This essentially identifies and selects a sampled 
    locations within a given radius (rand\\_flag*sqrt(2)) of packing location.

    Packing coordinates..
    - fill a rectangulare that is centered on 0,0.
    - x\\_dimension is normalized to 1
    - y\\_dimension is normalized (and assumed less than x\\_dimension)

    Therefore packing coords need to be scaled to rectangular area of the sampled
    locations.  Then find closest sampled location to each scaled packing coords.
    Note the assumption is sampled area has rectangular shape oriented with axes
    along x and y dimension.  If rotated off axis then will distort resulting
    locations.

    Return the index for these locations.

    See http://packomania.com/
"""
function periodic_sampling(
        percentage_samples::Float64,
        num_total::Integer,
        locs::DataFrames.DataFrame,
        pack_locs_in::GroupedDataFrame{DataFrames.DataFrame};
        #optional parameters
        rand_float::Float64=0.0, 
        group_flag::Bool=false)


    #how many indices?
    sample_points = round(Int, percentage_samples*num_total)
    #check that pack_locs has appropriate size group
    temp=[0]
    for idx in keys(pack_locs_in) push!(temp, idx[1][1]) end
    if !any(temp .== sample_points) # if a match, then make false since good
        error("A optimal packing spheres file doesn't exist for given number of points: "*string(sample_points))
    end
    #get specific group that we care about (change frome subDataFrame)
    pack_locs = DataFrames.DataFrame(pack_locs_in[(sample_points, )])

    #get dimensions of locs
    x_width  = abs(maximum(locs[:, :x])-minimum(locs[:, :x]))
    y_width  = abs(maximum(locs[:, :y])-minimum(locs[:, :y]))
    scale_width = x_width
    
    #get center of locs
    x_center = (maximum(locs[:, :x])-minimum(locs[:, :x]))/2.0 + minimum(locs[:, :x])
    y_center = (maximum(locs[:, :y])-minimum(locs[:, :y]))/2.0 + minimum(locs[:, :y])

    #flip x,y for pack_locs?  yes if y_width is greater than x_width
    x_width > y_width ? flip_xy=false : flip_xy=true

    #scale, shift, and apply rand offset to pack_locs
    if flip_xy
        rename!(pack_locs, :x=>:y, :y=>:x)
        scale_width = y_width
    end

    if group_flag # if true, get transformed location with no offset
      transform!(pack_locs, :x => (x-> x.*scale_width.+x_center) => :x)
      transform!(pack_locs, :y => (y-> y.*scale_width.+y_center) => :y)
    else          # if false, get transformed location with random offset
      transform!(pack_locs, :x => (x-> x.*scale_width.+x_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :x)
      transform!(pack_locs, :y => (y-> y.*scale_width.+y_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :y)
    end


    #now find the closest sampled location to each packing location
    sampling_idx = Int[]
    #outer loop over pack_locs
    for idx_pack in range(1, sample_points, step=1)
        temp_pack = Vector(pack_locs[idx_pack, [:x, :y]])
        min_idx = 1
        min_dist= Inf
        #inner loop over all possible sample locations
        for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(temp_pack - Vector(locs[idx_locs, [:x, :y]]))

            #get single location or grouping of locs around packing locs?
            if temp_dist < min_dist
              min_dist = temp_dist
              min_idx = idx_locs
            end
        end

        #now that found closest training sample to packed location
        #search over traiing samples again to group if flag is set
        if group_flag
          #search over all training samples 
          for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(Vector(locs[min_idx,[:x, :y]]) - Vector(locs[idx_locs, [:x, :y]]))

            if temp_dist < sqrt(2)*rand_float
              push!(sampling_idx, idx_locs)
            end
          end #idx_locs loop
        #else simply save sampling idx
        else
          push!(sampling_idx, min_idx)
        end #group_flag loop
    end #idx_pack loop

    return sampling_idx
end;

"""
    periodic\\_sampling(s\\_int, t\\_int, locs\\_dataframe, pack\\_coords\\_dataframe, rand\\_float)

    Periodically sample the available locations based on max packing concept, i.e.,
    coordinates are chosen that maximums radius between sampled locations.  Assumes
    that locs\\_dataframe is all the possible sampled locations within a rectangular
    area.  pack\\_coords are the coordinates for maximum packing given in normalized
    units, see http://packomania.com/.

    s\\_int is number of values to sample. t\\_int is used to bound-check (s\\_int < t\\_int)

    locs is assumed to be Nx2 dataframe with header names [:x, :y]

    pack\\_coords is assumped to be a grouped dataframe.  Each group is based on
    number of coords, e.g., groups 9, 18, 35, 71.

    rand\\_float sets the maximum uniformly distributed random offset. This parameter
    is used to address possibility that chosen packing location is a particularly
    bad set of measurements

    Packing coordinates..
    - fill a rectangulare that is centered on 0,0.
    - x\\_dimension is normalized to 1
    - y\\_dimension is normalized (and assumed less than x\\_dimension)

    Therefore packing coords need to be scaled to rectangular area of the sampled
    locations.  Then find closest sampled location to each scaled packing coords.
    Return the index for these locations.

    See http://packomania.com/
"""
function periodic_sampling(
        sample_points::Integer,
        num_total::Integer,
        locs::DataFrames.DataFrame,
        pack_locs_in::GroupedDataFrame{DataFrames.DataFrame};
        #optional parameters
        rand_float::Float64=0.0)


    #how many indices?
    if sample_points > num_total 
      error("num_samples ("*string(sample_points)*") is greater than sample size ("*string(num_total)*")")
    end

    #check that pack_locs has appropriate size group
    temp=[0]
    for idx in keys(pack_locs_in) push!(temp, idx[1][1]) end
    if !any(temp .== sample_points) # if a match, then make false since good
        error("A optimal packing spheres file doesn't exist for given number of points: "*string(sample_points))
    end
    #get specific group that we care about (change frome subDataFrame)
    pack_locs = DataFrames.DataFrame(pack_locs_in[(sample_points, )])

    #get dimensions of locs
    x_width  = abs(maximum(locs[:, :x])-minimum(locs[:, :x]))
    y_width  = abs(maximum(locs[:, :y])-minimum(locs[:, :y]))
    scale_width = x_width
    
    #get center of locs
    x_center = (maximum(locs[:, :x])-minimum(locs[:, :x]))/2.0 + minimum(locs[:, :x])
    y_center = (maximum(locs[:, :y])-minimum(locs[:, :y]))/2.0 + minimum(locs[:, :y])

    #flip x,y for pack_locs?  yes if y_width is greater than x_width
    x_width > y_width ? flip_xy=false : flip_xy=true

    #scale, shift, and apply rand offset to pack_locs
    if flip_xy
        rename!(pack_locs, :x=>:y, :y=>:x)
        scale_width = y_width
    end
    transform!(pack_locs, :x => (x-> x.*scale_width.+x_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :x)
    transform!(pack_locs, :y => (y-> y.*scale_width.+y_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :y)


    #now find the closest sampled location to each packing location
    sampling_idx = Int[]
    #outer loop over pack_locs
    for idx_pack in range(1, sample_points, step=1)
        temp_pack = Vector(pack_locs[idx_pack, [:x, :y]])
        min_idx = 1
        min_dist= Inf
        #inner loop over all possible sample locations
        for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(temp_pack - Vector(locs[idx_locs, [:x, :y]]))
            #get new index if distance is smaller
            if temp_dist < min_dist
                min_dist = temp_dist
                min_idx = idx_locs
            end
        end
        #save sampling idx
        push!(sampling_idx, min_idx)
    end

    return sampling_idx
end;

