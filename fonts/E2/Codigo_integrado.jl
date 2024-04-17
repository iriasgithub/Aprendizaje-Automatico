#Este código usa la función trainClassANN del Ejercicio_2, sin la posibilidad de pasar
#conjunto de validación ni test

using DelimitedFiles
using Flux, Plots

include("Ejercicio_2.jl")

function separador_dir()
    if Sys.iswindows()
        return "\\"
    else
        return "/"
    end
end

sep = separador_dir()


ruta = pwd()
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

"""topologies = [[8], [10, 6], [50, 100, 50, 20]]
 

# Gráfico para datos no normalizados
plot_layout1 = plot(xlabel="Epoch", ylabel="Loss", title="Loss: Unnormalized Data")
for topology in topologies
    _, lossVector = trainClassANN(topology, (inputs, targets))
    plot!(lossVector, label=string(topology))
end

# Gráfico para datos normalizados
plot_layout2 = plot(xlabel="Epoch", ylabel="Loss", title="Loss: Normalized Data")
for topology in topologies
    _, lossVector = trainClassANN(topology, (normalized_inputs, targets))
    plot!(lossVector, label=string(topology))
end

# Gráfico con learningRate alto
plot_layout3 = plot( xlabel="Epoch", ylabel="Loss", title="Loss: High Learning Rate")
for topology in topologies
    _, lossVector = trainClassANN(topology, (normalized_inputs, targets), learningRate=0.5)
    plot!(lossVector, label=string(topology))
end

plot(plot_layout1, plot_layout2, plot_layout3)"""

#Conclusiones.
#- Una tasa de aprendizaje alta con una topología excesivamente compleja hace que la red no converja. Sobreajuste.
#- Datos no normalizados producen una connvergencia más lenta.
#- Datos no normalizados con topología compleja producen anomalías en la convergencia en algunos epochs.
ann, lossVector = trainClassANN([3], (normalized_inputs, targets))
outputs = ann(normalized_inputs')'
cod_outputs = classifyOutputs(outputs)
println(cod_outputs)
println(targets)
println(typeof(cod_outputs))
println(outputs)
println(typeof(targets))
accuracy(outputs, targets)