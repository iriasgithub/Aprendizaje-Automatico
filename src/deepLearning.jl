using Flux
using Plots
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean

include("adhoc.jl")
include("arquitectures.jl")

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


#Cargar base de datos

no_melanoma_color = cargar_imagenes("no_melanoma", "dermoscopic", false)
melanoma_color = cargar_imagenes("melanoma", "dermoscopic", false)
atypical_color = cargar_imagenes("atypical_nevus", "dermoscopic", false)

instances = length(no_melanoma_color) + length(melanoma_color) + length(atypical_color)
index = shuffle(1:instances)
inputs_color, targets = crear_inputs_targets_multiclase(no_melanoma_color, melanoma_color, atypical_color, index)

ruta_destino = "resized_images"*sep
ruta_destino2 = "images"*sep

#Resize imagenes al mismo tamaño
#=
for i in 1:length(inputs_color)
    ruta_completa1 = joinpath(ruta_destino2, string(i)*".png")
    save(ruta_completa1, inputs_color[i])
    inputs_color[i] = imresize(inputs_color[i], (30, 30))
    ruta_completa = joinpath(ruta_destino, string(i)*".png")
    save(ruta_completa, inputs_color[i])
end
=#

inputs_color = [imresize(img, (30, 30)) for img in inputs_color]


#Hold out para dividir en train y test porque no se realiza validación cruzada
index_train, index_test = holdOut(length(inputs_color), 0.2)

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float32, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float32,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float32,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float32,2}, blue.(image));
    return matrix;
end;


train_imgs = imageToColorArray.(inputs_color[index_train])
test_imgs = imageToColorArray.(inputs_color[index_test])
train_labels = targets[index_train]
test_labels = targets[index_test]


function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 30, 30, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(30,30,3)) "Las imagenes no tienen tamaño 50x50";
        nuevoArray[:,:,:,i] .= imagenes[i];
    end;
    return nuevoArray;
end;

train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

labels = unique(targets)
train_set = (train_imgs, onehotbatch(train_labels, labels))
test_set = (test_imgs, onehotbatch(test_labels, labels))

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))

println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");


function train(ann)
    
    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    L1 = 0.001
    L2 = 0.001

    absnorm(x) = sum(abs , x)
    sqrnorm(x) = sum(abs2, x)
    loss(model,x,y) = ((size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y)) + L1*sum(absnorm, Flux.params(model)) + L2*sum(sqrnorm, Flux.params(model));
    eta = 0.01;
    opt_state = Flux.setup(Adam(eta), ann);

    println("Comenzando entrenamiento con arquitectura ", ann)
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;

    while !criterioFin       
    
        # Se entrena un ciclo
        Flux.train!(loss, ann, [train_set], opt_state);
    
        numCiclo += 1;
    
        # Se calcula la precision en el conjunto de entrenamiento:

        (_, _, _, _, _, _, precisionEntrenamiento, _) = confusionMatrix(ann(train_set[1])', train_set[2]')
        #println("Ciclo ", numCiclo, ": F1 en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");
    
        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento > mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            #precisionTest = mean(onecold(ann(test_set[1])) .== onecold(test_set[2]));
            (_, _, _, _, _, _, precisionTest, _) = confusionMatrix(ann(test_set[1])', test_set[2]');
            #println("   Mejora en el conjunto de entrenamiento -> F1 en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end
        
        # Criterios de parada:
    
        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a un F1 de 99.9%")
            criterioFin = true;
            return precisionTest, precisionEntrenamiento
        end
    
        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 20)
            (_, _, _, _, _, _, precisionTest, _) = confusionMatrix(ann(test_set[1])', test_set[2]');
            println("   Se para el entrenamiento por no haber mejorado el F1 en el conjunto de entrenamiento durante 20 ciclos con un F1 en test de ", 100*precisionTest, " % y en
            entrenamiento de ", 100*precisionEntrenamiento, " %")
            criterioFin = true;
            return precisionTest, precisionEntrenamiento
        end
        
    end
    
end

f1_score_test = []
f1_score_training = []

for ann in arquitecturas
    result = train(ann)
    push!(f1_score_test, result[1])
    push!(f1_score_training, result[2])
end

println(f1_score_test)
println(f1_score_training)

x = 1:length(arquitecturas)

bar(x, [f1_score_test f1_score_training], label=["F1 Score Test" "F1 Score Training"], xlabel="Arquitectura", ylabel="F1 Score", title="Comparación F1 Score en Conjunto de Test y Training", xticks=(1:length(arquitecturas)))