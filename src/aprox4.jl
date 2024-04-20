using Plots
using Random

include("adhoc.jl")
include("plotting.jl")

Random.seed!(123)

no_melanoma_bmp = cargar_imagenes("no_melanoma", "lesion", true)
melanoma_bmp = cargar_imagenes("melanoma", "lesion", true)
atypical_bmp = cargar_imagenes("atypical_nevus", "lesion", true)

no_melanoma_gray = cargar_imagenes("no_melanoma", "dermoscopic", true)
melanoma_gray = cargar_imagenes("melanoma", "dermoscopic", true)
atypical_gray = cargar_imagenes("atypical_nevus", "dermoscopic", true)

no_melanoma_color = cargar_imagenes("no_melanoma", "dermoscopic", false)
melanoma_color = cargar_imagenes("melanoma", "dermoscopic", false)
atypical_color = cargar_imagenes("atypical_nevus", "dermoscopic", false)

instances = length(melanoma_bmp) + length(no_melanoma_bmp) + length(atypical_bmp)
index = shuffle(1:instances)
inputs_bmp, _ = crear_inputs_targets_multiclase(no_melanoma_bmp, melanoma_bmp, atypical_bmp, index)
inputs_gray, _ = crear_inputs_targets_multiclase(no_melanoma_gray, melanoma_gray, atypical_gray, index)
inputs_color, targets = crear_inputs_targets_multiclase(no_melanoma_color, melanoma_color, atypical_color, index)

# Creación del vector de las coordenadas de las bounding boxes de cada imagen
axis = Vector{Any}(undef, size(inputs_bmp, 1))
# Creamos las imágenes del bounding box y las guardamos en una carpeta
axis .= create_bounding_box.(inputs_bmp)

dataset = extraer_caracteristicas_aprox3(inputs_bmp, inputs_gray, inputs_color, axis)
dataset = Float32.(dataset)
println(size(dataset))
normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

#Plotteado de los datos
datos_no_melanoma = dataset[targets .== "no_melanoma", :]
datos_melanoma = dataset[targets .== "melanoma", :]
datos_atypical_nevus = dataset[targets .== "atypical_nevus", :]

scatter3d(datos_no_melanoma[:, 1], datos_no_melanoma[:, 2], datos_no_melanoma[:, 3], 
    xlabel="Longitud del Borde", ylabel="Asimetría", zlabel="Claridad",
    color=:green, label="No Melanoma")

scatter3d!(datos_melanoma[:, 1], datos_melanoma[:, 2], datos_melanoma[:, 3] , 
xlabel="Longitud del Borde", ylabel="Asimetría", zlabel="Claridad",
color=:red, label="Melanoma")

scatter3d!(datos_atypical_nevus[:, 1], datos_atypical_nevus[:, 2], datos_atypical_nevus[:, 3] , 
xlabel="Longitud del Borde", ylabel="Asimetría", zlabel="Claridad",
color=:blue, label="Atypical Nevus")

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
profundidades = [16, 20, 24, 28, 32, 36, 40, 44]

(max_mean_f1_ARB, best_depth) = execute_ARB(profundidades, dataset, targets)
println(max_mean_f1_ARB, best_depth)
########################################################################################################################

# Ejecución de KNN
K = [1, 3, 5, 7, 9, 11, 13, 15, 17]

(max_mean_f1_KNN, best_k) = execute_KNN(K, dataset, targets)
println(max_mean_f1_KNN, best_k)
########################################################################################################################

#Ejecución de ANN
topologias = [[2], [4], [8], [2, 2], [4, 2], [4, 4], [8, 4], [8, 8]]

(max_mean_f1_ANN, best_topology) = execute_ANN(topologias, dataset, targets)
println(max_mean_f1_ANN, best_topology)
########################################################################################################################