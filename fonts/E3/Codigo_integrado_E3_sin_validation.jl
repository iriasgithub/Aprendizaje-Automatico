using DelimitedFiles
using Flux

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
dataset = readdlm(ruta_completa, ',')

inputs = dataset[:, 1:4]
targets = dataset[:, 5]

#Codificamos las entradas y salidas al formato correcto
inputs = convert(Array{Float32, 2}, inputs)
targets = oneHotEncoding(targets)

@assert (size(inputs,1) == size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

# Normalizamos los datos de entrada
normalized_inputs = normalizeMinMax(inputs)

(ann, v_train_loss, v_val_loss, v_test_loss) = trainClassANN([3, 10], (normalized_inputs, targets))

outputs = ann(normalized_inputs')'
cod_outputs = classifyOutputs(outputs)
println(accuracy(outputs, targets))