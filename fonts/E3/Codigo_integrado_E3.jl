using DelimitedFiles
using Flux, Plots

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

include(ejercicio_2)
include("Ejercicio_3.jl")

ruta_completa = ruta*"$sep"*"fonts"*"$sep"*"E2"*"$sep"*"iris_database"*"$sep"*"iris.data"

#Cargar la base de datos con patrones en filas y atributos y salidas deseadas en columnas.
dataset = readdlm(ruta_completa, ',')

inputs = dataset[:, 1:4]
targets = dataset[:, 5]

#Codificamos las entradas y salidas al formato correcto
inputs = convert(Array{Float32, 2}, inputs)
targets = oneHotEncoding(targets)

(trainSet, validationSet, testSet) = holdOut(size(dataset, 1), 0.3, 0.3)

#Indexamos solo las filas del trainSet
train_inputs = inputs[trainSet, :]
train_targets = targets[trainSet, :]

val_inputs = inputs[validationSet, :]
val_targets = targets[validationSet, :]

test_inputs = inputs[testSet, :]
test_targets = targets[testSet, :]

#Calculamos los valores del min max de los atributos del conjunto de entrenamiento
normalization_parameters_min_max = calculateMinMaxNormalizationParameters(train_inputs)

normalizeMinMax!(train_inputs, normalization_parameters_min_max)
normalizeMinMax!(val_inputs, normalization_parameters_min_max)
normalizeMinMax!(test_inputs, normalization_parameters_min_max)

trainingDataset = (train_inputs, train_targets)
validationDataset = (val_inputs, val_targets)
testDataset = (test_inputs, test_targets)

topologies = [[8], [4, 4], [16, 8, 4]]

plots = []

for topology in topologies
    (_, v_train_loss, v_val_loss, v_test_loss) = trainClassANN(topology, trainingDataset; validationDataset, testDataset)
    g = plot(xaxis = "Epoch", yaxis = "Loss", title = "Topology: " * string(topology))
    plot!(v_train_loss, label = "train")
    plot!(v_val_loss, label = "validation")
    plot!(v_test_loss, label = "test")
    push!(plots, g)
end

display(plot(plots..., layout = (3,1)))
