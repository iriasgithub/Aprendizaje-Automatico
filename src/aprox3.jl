using Plots
using Random

include("adhoc.jl")
include("plotting.jl")

Random.seed!(123)

no_melanoma_bmp = cargar_imagenes("no_melanoma", "lesion", true)
melanoma_bmp = cargar_imagenes("melanoma", "lesion", true)
no_melanoma_gray = cargar_imagenes("no_melanoma", "dermoscopic", true)
melanoma_gray = cargar_imagenes("melanoma", "dermoscopic", true)
no_melanoma_color = cargar_imagenes("no_melanoma", "dermoscopic", false)
melanoma_color = cargar_imagenes("melanoma", "dermoscopic", false)

instances = length(melanoma_bmp) + length(no_melanoma_bmp)
index = shuffle(1:instances)
inputs_bmp, _ = crear_inputs_targets_binarias(no_melanoma_bmp, melanoma_bmp, index)
inputs_gray, _ = crear_inputs_targets_binarias(no_melanoma_gray, melanoma_gray, index)
inputs_color, targets = crear_inputs_targets_binarias(no_melanoma_color, melanoma_color, index)


# Creación del vector de las coordenadas de las bounding boxes de cada imagen
axis = Vector{Any}(undef, size(inputs_bmp, 1))
# Creamos las imágenes del bounding box y las guardamos en una carpeta
axis .= create_bounding_box.(inputs_bmp)

dataset = extraer_caracteristicas_aprox3(inputs_bmp, inputs_gray, inputs_color, axis)
dataset = Float32.(dataset)
normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

#Plotteado de los datos
datos_clase_0 = dataset[targets .== 0, :]
datos_clase_1 = dataset[targets .== 1, :]

scatter(datos_clase_0[:, 4], datos_clase_0[:, 2], xlabel = "Color", ylabel = "Asimetría", color=:green, label="No melanoma")
scatter!(datos_clase_1[:, 4], datos_clase_1[:, 2], xlabel = "Color", ylabel = "Asimetría", color=:red, label="Melanoma")

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