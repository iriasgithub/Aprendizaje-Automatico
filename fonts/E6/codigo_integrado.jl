include("Ejercicio_6.jl")

using DelimitedFiles
using Flux, Plots

#Función para obtener el separador del sistema
function separador_dir()
    if Sys.iswindows()
        return "\\"
    else
        return "/"
    end
end

sep = separador_dir()

ruta = pwd()
ejercicio_2 = ruta*"$sep"*"fonts"*"$sep"*"E2"*"$sep"*"Ejercicio_2.jl"
ejercicio_3 = ruta*"$sep"*"fonts"*"$sep"*"E3"*"$sep"*"Ejercicio_3.jl"
ejercicio_5 = ruta*"$sep"*"fonts"*"$sep"*"E5"*"$sep"*"Ejercicio_5.jl"

include(ejercicio_2)
include(ejercicio_3)
include(ejercicio_5)

ruta_completa = ruta*"$sep"*"fonts"*"$sep"*"E2"*"$sep"*"iris_database"*"$sep"*"iris.data"

#Cargar la base de datos con patrones en filas y atributos y salidas deseadas en columnas.
dataset = readdlm(ruta_completa, ',')

inputs = dataset[:, 1:4]
targets = dataset[:, 5]

#Codificamos las entradas y salidas al formato correcto
inputs = convert(Array{Float32, 2}, inputs)

topology = [16, 8, 4]
modelHypANN = Dict()
modelHypANN["topology"] = topology
modelHypANN["numExecutions"] = 10

modelCrossValidation(:ANN, modelHypANN, inputs, targets, crossvalidation(targets, 10))

#=
modelHypSVC = Dict()
modelHypSVC["C"] = 1.0
modelHypSVC["kernel"] = "rbf"
modelHypSVC["degree"] = 3
modelHypSVC["gamma"] = "scale"
modelHypSVC["coef0"] = 0.0

modelCrossValidation(:SVC, modelHypSVC, inputs, targets, crossvalidation(targets, 10))


modelHypTree = Dict()
modelHypTree["max_depth"] = 3

modelCrossValidation(:DecisionTreeClassifier, modelHypTree, inputs, targets, crossvalidation(targets, 10))


modelHypKNN = Dict()
modelHypKNN["n_neighbors"] = 3

modelCrossValidation(:KNeighborsClassifier, modelHypKNN, inputs, targets, crossvalidation(targets, 10))
=#
