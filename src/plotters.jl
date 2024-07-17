# COV_EXCL_START
using KernelAbstractions
using StaticArrays
using CairoMakie
function pointidx(i::Integer, j::Integer, order)
    # I'm sorry
    # Quad: https://gitlab.kitware.com/nick.laurenson/vtk/-/blob/d8aa5b89c622cf04b3322112fda420afc9d2a16d/Common/DataModel/vtkLagrangeQuadrilateral.cxx#L558
    # Hex: https://gitlab.kitware.com/nick.laurenson/vtk/blob/d8aa5b89c622cf04b3322112fda420afc9d2a16d/Common/DataModel/vtkLagrangeHexahedron.cxx#L734
    # Implemented for quad here
    is_i_boundary = i == 0 || i == order[1]
    is_j_boundary = j == 0 || j == order[2]

    n_boundary = is_i_boundary + is_j_boundary

    i_pos = i > 0
    j_pos = j > 0

    if n_boundary == 2
        # Vertex DOF
        return (i_pos ? (j_pos ? 2 : 1) : (j_pos ? 3 : 0))
    end

    offset = 4
    if n_boundary == 1
        # Edge DOF
        if !is_i_boundary
            # On i axis
            return (i - 1) +
                   (j_pos ? order[1] - 1 + order[2] - 1 : 0) +
                   offset
        end
        if !is_j_boundary
            # On j axis
            return (j - 1) +
                   (i_pos ? order[1] - 1 : 2 * (order[1] - 1) + order[2] - 1) +
                   offset
        end
    end

    offset += 2 * (order[1] - 1 + order[2] - 1)
    # n_boundary == 0 -> Face DOF
    return offset +
           (i - 1) + (order[1] - 1) * (
               (j - 1))


end

function get_plot_points(order)
    dim = 2
    n_points_x = (order + 1)
    n_points = (order + 1) * (order + 1)
    n_points_x, n_points
    points = zeros(dim, n_points)
    #xs = range(0.0, 1.0, length=n_points_x)
    xs = (gausslobatto(n_points_x)[1] .+ 1) ./ 2
    ys = xs
    for xidx = 1:n_points_x
        for yidx = 1:n_points_x
            idx = pointidx(xidx - 1, yidx - 1, (order, order)) + 1
            x = xs[xidx]
            y = ys[yidx]
            points[:, idx] = [x, y]
        end
    end
    return transpose(points), size(points, 2)
end


"""
    VTKPlotter(eq::Equation, scenario::Scenario, grid::Grid,
               filename::String)

Initialize a VTKPlotter for equation `eq` and scenario `scenario`,
defined on `grid_gpu`. The gpu grid is copied to cpu for plotting.
Output name is `filename`.
"""
mutable struct VTKPlotter{F}
    eq::Equation
    scenario::Scenario
    filename::String
    collection::WriteVTK.CollectionFile
    plot_counter::Int64
    vtkcells::Array{MeshCell,1}
    vtkpoints::Array{Float64,2}
    cellpoints_gpu::AbstractArray{F,2}
    plot_tasks::Vector{Task}
    save_images::Bool

    function VTKPlotter(eq::Equation, scenario::Scenario, grid_gpu::Grid{F,T1,T2,2,ndofs,order,T3},
        filename::String, save_images::Bool) where {F,T1,T2,ndofs,order,T3}
        grid = cpu(grid_gpu)
        collection = paraview_collection(filename)
        plot_counter = 0

        cellpoints, num_points_per_cell = get_plot_points(order)

        vtkcells = Array{MeshCell,1}(undef, length(grid.cells))
        vtkpoints = Array{F,2}(undef, (2, length(grid.cells) * num_points_per_cell))

        for (i, cell) in enumerate(grid.cells)
            offset = cell.center .- (cell.size[1] / 2, cell.size[2] / 2)
            start = (i - 1) * num_points_per_cell
            for j = 1:num_points_per_cell
                vtkpoints[:, start+j] = offset .+ cellpoints[j, :] .* cell.size
            end
            vtkcells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, Array(start+1:start+num_points_per_cell))
        end
        cellpoints_gpu = similar(grid_gpu.dofs, size(cellpoints))
        copyto!(cellpoints_gpu, collect(cellpoints))


        new{F}(eq, scenario, filename, collection, plot_counter, vtkcells, vtkpoints, cellpoints_gpu, [], save_images)
    end

end

"""
    evaluate_dof_points(grid::Grid, points, dofidx)

Evaluate the DOF points for `dofidx` on `grid` at `points`, potentially on GPU.

Returns the evaluated data of size `(length(grid.cells)*num_points, 1)`.
"""
function evaluate_dof_points(grid::TerraDG.Grid{F,T1,T2,2,ndofs,order,T3}, points, dofidx) where {F,T1,T2,ndofs,order,T3}
    num_points = size(points, 1)
    eval_data = similar(grid.dofs, (length(grid.cells) * num_points, 1))
    basis = grid.basis
    dofs = grid.dofs
    @kernel function evaluate_cell_kernel()
        cell_id = @index(Global)
        offset = (cell_id - 1) * num_points
        celldofs = SVector{order^2,F}(@view(dofs[cell_id, 1:order^2, dofidx]))
        for i = 1:num_points
            point = SVector{2,F}(@view(points[i, 1:2]))
            eval_data[offset+i, :] .= evaluate_basis(basis, celldofs, point)
        end
    end
    backend = get_backend(dofs)
    kernel = evaluate_cell_kernel(backend)
    kernel(ndrange=length(grid.cells))
    KernelAbstractions.synchronize(backend)
    eval_data
end


"""
    plot(plotter::VTKPlotter)

Write output with `plotter` for timestep.

This function is asynchronous and returns immediately after copying the grid.
"""
function plot(plotter::VTKPlotter, grid_gpu::Grid)
    @info "started copying grid for plotting for t=$(grid_gpu.time)"
    grid = deepcopy(grid_gpu)
    @info "finished copying grid for plotting for t=$(grid_gpu.time)"
    count_to_plot = plotter.plot_counter
    plotter.plot_counter += 1
    t = Threads.@spawn begin
        # Evaluate the dofs at all points in a separate thread
        eq = plotter.eq
        currentfile = @sprintf("%s_%d", plotter.filename, count_to_plot)
        vtkfile = vtk_grid(currentfile, plotter.vtkpoints, plotter.vtkcells)

        @info "start writing data for t=$(grid.time)"
        for var = 1:get_ndofs(eq)
            t = time()
            evaluated_data = evaluate_dof_points(grid, plotter.cellpoints_gpu, var)
            dt = time() - t
            @info "evaluated data for var=$(var) in $(dt) seconds"

            cpu_data = Array(evaluated_data)
            vtk_point_data(vtkfile, cpu_data, string(get_variable_name(eq, var)))
            dt = time() - t
            t = time()
            @info "wrote data to disk for var=$(var) in $dt seconds"
            # plot to image
            if plotter.save_images
                x = plotter.vtkpoints[1, :]
                y = plotter.vtkpoints[2, :]
                pointdata = cpu_data[:, 1]
                all_data_vector = zip(x, y, pointdata)
                all_data_vector = unique(p -> p[1:2], all_data_vector)
                x = [all_data_vector[i][1] for i in 1:length(all_data_vector)]
                y = [all_data_vector[i][2] for i in 1:length(all_data_vector)]
                data = [all_data_vector[i][3] for i in 1:length(all_data_vector)]
                f = Figure()
                ax = Axis(f[1, 1], xlabel = "x", ylabel = "y", title = "$(get_variable_name(eq, var)) t=$(grid.time)")
                hm = heatmap!(ax, x, y, data, colormap = :viridis)
                Colorbar(f[:, end+1], hm)
                folder = "plots/$(get_variable_name(eq, var))"
                mkpath(folder)
                Makie.save("$folder/$(grid.time).png", f)
                @info "saved image for var=$(var) in $(time()-t) seconds"
            end
        end
        @info "finished writing data for t=$(grid.time)"
        collection_add_timestep(plotter.collection, vtkfile, grid.time)
        @info "Added to collection for t=$(grid.time)"
    end
    push!(plotter.plot_tasks, t)
end

"""
    save(plotter::VTKPlotter)

Save final output file for `plotter`.
"""
function save(plotter::VTKPlotter)
    @info "Waiting for all tasks to finish"
    for t in plotter.plot_tasks
        fetch(t)
    end
    @info "All tasks finished"
    vtk_save(plotter.collection)
    @info "Saved collection"
end
# COV_EXCL_STOP