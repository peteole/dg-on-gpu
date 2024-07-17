using Revise
using TerraDG


configfile = "src/input/euler.yaml"
#configfile = "src/input/advection.yaml"
TerraDG.main(configfile)
