
using Plots
using Random

include("adhoc.jl")
include("plotting.jl")

Random.seed!(123)

no_melanoma_imgs = cargar_imagenes("no_melanoma", "lesion", true)
melanoma_imgs = cargar_imagenes("melanoma", "lesion", true)

instances = length(melanoma_imgs) + length(no_melanoma_imgs)
index = shuffle(1:instances)
inputs, targets = crear_inputs_targets_binarias(no_melanoma_imgs, melanoma_imgs, index)

# Creación del vector de las coordenadas de las bounding boxes de cada imagen
axis = Vector{Any}(undef, size(inputs, 1))
# Creamos las imágenes del bounding box y las guardamos en una carpeta
axis .= create_bounding_box.(inputs)

dataset = extraer_caracteristicas_aprox1(inputs, axis)
dataset = Float32.(dataset)
normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

#Plotteado de los datos
datos_clase_0 = dataset[targets .== 0, :]
datos_clase_1 = dataset[targets .== 1, :]
scatter(datos_clase_0[:, 1], datos_clase_0[:, 2], xlabel = "Regularidad", ylabel = "Asimetría", color=:green, label="Clase 0")
scatter!(datos_clase_1[:, 1], datos_clase_1[:, 2], xlabel = "Regularidad", ylabel = "Asimetría", color=:red, label="Clase 1")
########################################################################################################################

#Ejecución de SVM
C_val = [0.1, 1, 10, 100]
kernel_val = ["linear", "poly", "rbf" , "sigmoid"]
producto_cartesiano = Iterators.product(C_val, kernel_val)
# Convertir el producto cartesiano en una lista de tuplas
configuraciones = collect(producto_cartesiano)

(max_mean_f1_SVM, best_config) = execute_SVM(configuraciones, dataset, targets)
println(max_mean_f1_SVM, best_config)
########################################################################################################################

#Ejecución de DT
profundidades = [2, 3, 4, 5, 8, 9, 16]

(max_mean_f1_ARB, best_depth) = execute_ARB(profundidades, dataset, targets)
println(max_mean_f1_ARB, best_depth)
########################################################################################################################

# Ejecución de KNN
K = [1, 3, 5, 7, 9, 11]

(max_mean_f1_KNN, best_k) = execute_KNN(K, dataset, targets)
println(max_mean_f1_KNN, best_k)
########################################################################################################################

#Ejecución de ANN
topologias = [[2], [4], [8], [2, 2], [4, 2], [4, 4], [8, 4], [8, 8]]

(max_mean_f1_ANN, best_topology) = execute_ANN(topologias, dataset, targets)
println(max_mean_f1_ANN, best_topology)
########################################################################################################################