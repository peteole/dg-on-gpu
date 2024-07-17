# TerraDG.jl on the GPU

[![Euler equations solution](http://img.youtube.com/vi/3WDuZH6MymY/0.jpg)](http://www.youtube.com/watch?v=3WDuZH6MymY "Euler equations solution")

This is a solution of the Euler equations using the TerraDG.jl package on the GPU. The video shows the density of the fluid at different time steps. The simulation is run on a 500x500 grid with a fourth-order solver.

### Running the code

- Clone the repo
- Start Julia in the root directory of the project with `julia --project=.`. Optionally, you can set the number of threads with `export JULIA_NUM_THREADS=8`.
- Run `] instantiate` to install the dependencies.
- Run `include("src/main.jl")` to run the code. You can change the parameters in the `main.jl` file. Note that the increased complexity and number of dependencies of the project increased precompilation time, sometimes taking up to around 3 minutes.

In addition to the parameters from the worksheets, you can specify the following parameters in the input file:

- `solver.slope_limiting: true`: Activate minmod slope limiting.
- `simulation.device: cuda:3`: Select the device to run the simulation on. The default is `cpu`. Allowed values are `cpu`,`cuda`, `cuda:X`, where `X` is the device number, `amdgpu`, `amdgpu:X`, where `X` is the device number.
- `simulation.float_type: Float32`: Select the floating point precision. Allowed values are `Float32` and `Float64`.
- `output.save_images: true`: Save plots of each dof in the `plots` folder.
- `output.log_level: warn`: Set the log level. Allowed values are `debug`, `info`, `warn`, `error`.

### Running the code with Docker

- Build the Docker image with `docker build -t terradg .`.
- Run the Docker container with `docker run -it terradg`. This will start an interactive Julia session in the TerraDG project.
- Run your experiment using `using TerraDG; TerraDG.main("src/input/advection.yaml")`.

If you also wish to play with the inputs and inspect the plots, you can mount the`src/input`, `output` and `plots` directories to your host machine. First create the folders if they do not exist: `mkdir plots;mkdir output`. For example, run `docker run -it -v $(pwd)/src/input:/app/src/input -v $(pwd)/output:/app/output -v $(pwd)/plots:/app/plots terradg`. Note that GPU acceleration is not available in the Docker container.

### Folder structure

- `src`: Contains the source code of the project.
- `results`: Contains documentation for all worksheets
- `benchmark`: Contains benchmarking scripts and results
- `output`: Contains the output of the simulation in vtk format. Can be opened with Paraview.
- `plots`: Contains plots of the simulation. Can be disabled in the input file.
- `test`: Contains the tests for the project.
- `worksheets.ipynb`: Contains the output of all worksheets in a Jupyter notebook. Running this notebook runs all the worksheet configs. In the `cuda_video_output` branch, this file contains the output as a video. However, we experienced compatibility issues.

### Branches and tags

- branch `main`: Contains the project implementation with all optimizations.
- tags `worksheet-X` (X=1,2,3,4): Contains the implementation of the respective worksheet.
- branch `cuda_video_output`: Contains the implementation of the video output for the Euler equations on the GPU in the `worksheets.ipynb` file.
- branch `cuda_derivative_factorization`: Contains the project but without the surface integral optimization. More flexible (we do not make assumptions about the order of the cells here) and readable code but around 30% slower.

### Testing

Run `] test` to run the tests.

```@contents
```

# Main
```@docs
TerraDG.evaluate_rhs
TerraDG.main
```
# I/O
## Configuration
```@docs
TerraDG.Configuration
TerraDG.get_device_array_type
```

## VTK/Paraview output
```@docs
TerraDG.VTKPlotter
TerraDG.plot
TerraDG.save
TerraDG.evaluate_dof_points
```

## Error Writer
```@docs
TerraDG.evaluate_error
```

# Equations
```@docs
TerraDG.Equation
TerraDG.make_equation
TerraDG.Scenario
TerraDG.make_scenario
TerraDG.interpolate_initial_dofs
TerraDG.get_nvars
TerraDG.get_nparams
TerraDG.get_variable_name
TerraDG.is_periodic_boundary
TerraDG.evaluate_boundary
TerraDG.get_initial_values
TerraDG.is_analytical_solution
TerraDG.evaluate_flux
TerraDG.max_eigenval
TerraDG.@declare_dofs
```

# Grid
```@docs
TerraDG.FaceType
TerraDG.Cell
TerraDG.Grid
TerraDG.Face
TerraDG.get_neighbor
TerraDG.make_mesh
TerraDG.make_grid
TerraDG.globalposition
TerraDG.volume
TerraDG.area
TerraDG.inverse_jacobian
TerraDG.cpu
```

## Basis
```@docs
TerraDG.lagrange_1d
TerraDG.lagrange_diff
TerraDG.get_quadpoints
TerraDG.Basis
Base.length(::TerraDG.Basis)
Base.size(::TerraDG.Basis)
Base.size(::TerraDG.Basis,::Integer)
Base.size
TerraDG.evaluate_basis
TerraDG.project_to_reference_basis
TerraDG.massmatrix
TerraDG.derivativematrix
TerraDG.get_face_quadpoints
TerraDG.face_projection_matrix
TerraDG.evaluate_m_to_n_vandermonde_basis
```

# Kernels
Many kernels take both global matrices and buffers.
Try to use them to avoid costly re-computations or memory
allocations.

## Global Matrices
```@docs
TerraDG.GlobalMatrices
```

## Time
```@docs
TerraDG.TimeIntegrator
TerraDG.make_timeintegrator
TerraDG.step
TerraDG.ExplicitEuler
TerraDG.SSPRK2
TerraDG.SSPRK3
```
## Surface
```@docs
TerraDG.project_to_face
TerraDG.evaluate_face_integrals
TerraDG.project_flux_to_face
TerraDG.project_face_to_inner
```

## Volume
```@docs
TerraDG.evaluate_volumes
```

## Flux
```@docs
TerraDG.evaluate_fluxes
```

## Slope limiting
```@docs
TerraDG.minmod_slope_limiting!
TerraDG.SlopeLimitingBuffer
TerraDG.compute_slope_limiting_coefficients!
```
## Initial Conditions
```@docs
TerraDG.set_initial_conditions
```

## Index
```@index
```
