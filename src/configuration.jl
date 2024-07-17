using Logging
using CUDA
"""
    Configuration(configfile::String)

Parses configuration file `configfile`.
"""
mutable struct Configuration{F<:Real}
    equation_name::String
    scenario_name::String

    end_time::F
    order::Int64
    timeintegrator_name::String
    courant::F

    filter_name::String
    filter_order::Int64
    plot_start::F
    plot_step::F
    grid_elements::Int64
    physicalsize::NTuple{2,F}
    cellsize::NTuple{2,F}
    slope_limiting::Bool
    device::String
    save_images::Bool

    function Configuration(configfile::String)
        config = YAML.load(open(configfile))
        equation_name = config["equation"]["equation"]
        scenario_name = config["equation"]["scenario"]
        float_type = get(config["simulation"], "float_type", "Float64")
        F = if float_type == "Float32"
            Float32
        elseif float_type == "Float64"
            Float64
        else
            error("Unknown float type: $float_type")
        end
        device = get(config["simulation"], "device", "cpu")
        save_images = get(config["output"], "save_images", false)

        log_level = get(config["output"], "log_level", "info")
        if log_level == "info"
            global_logger(ConsoleLogger(stderr, Logging.Info))
        elseif log_level == "warn"
            global_logger(ConsoleLogger(stderr, Logging.Warn))
        elseif log_level == "error"
            global_logger(ConsoleLogger(stderr, Logging.Error))
        else
            error("Unknown log level: $log_level")
        end

        end_time = config["simulation"]["end_time"]
        order = config["solver"]["order"]
        timeintegrator_name = config["solver"]["timeintegrator"]
        courant = config["solver"]["courant"]

        filter_name = get(config["solver"], "filter", "identity")
        filter_order = get(config["solver"], "filter_order", order)
        plot_start = config["output"]["start"]
        plot_step = config["output"]["step"]
        grid_elements = config["simulation"]["grid_elements"]
        physicalsize_1d = config["simulation"]["grid_size"]
        physicalsize = (physicalsize_1d, physicalsize_1d)
        cellsize = physicalsize ./ grid_elements
        slope_limiting = get(config["solver"], "slope_limiting", false)

        new{F}(
            equation_name,
            scenario_name,
            F(end_time),
            order,
            timeintegrator_name,
            courant,
            filter_name,
            filter_order,
            F(plot_start),
            F(plot_step),
            grid_elements,
            F.(physicalsize),
            F.(cellsize),
            slope_limiting,
            device,
            save_images
        )
    end
end

"""
Gets the array type for the device specified in the configuration.
Also sets the default device and dynamically loads the corresponding
accelerator module.
"""
function get_device_array_type(config::Configuration)
    if config.device == "cpu"
        return Array
    elseif startswith(config.device, "cuda")
        # match "cuda:0" or "cuda:1" etc.
        if !CUDA.functional()
            @warn "CUDA not functional, falling back to CPU"
            return Array
        end
        m = match(r"cuda:(\d+)", config.device)
        if m !== nothing
            device_id = parse(Int, first(m.captures))
            CUDA.device!(device_id)
        elseif config.device !== "cuda"
            error("Unknown device: $(config.device)")
        end
        return CuArray
    elseif startswith(config.device, "amdgpu")
        # avoid loading AMDGPU.jl if not necessary since it throws a warning if no amd gpu is installed
        @eval using AMDGPU
        @warn "AMDGPU was loaded dynamically, probably you have to run this function again within the same session"
        # match "opencl:0" or "opencl:1" etc.
        m = match(r"amdgpu:(\d+)", config.device)
        if m !== nothing
            device_id = parse(Int, first(m.captures))
            AMDGPU.device!(device_id)
        elseif config.device !== "amdgpu"
            error("Unknown device: $(config.device)")
        end
        return ROCArray
    # elseif config.device == "mps"
    #     return MtlArray
    else
        error("Unknown device: $(config.device)")
    end
end 