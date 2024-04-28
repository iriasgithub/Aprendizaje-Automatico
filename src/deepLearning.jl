using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean

Random.seed!(123)

include("adhoc.jl")

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

#Resize imagenes al mismo tamaño
inputs_color = [imresize(img, (570, 760)) for img in inputs_color]

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

println(test_labels)

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 570, 760, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(570,760,3)) "Las imagenes no tienen tamaño 570x760";
        nuevoArray[:,:,:,i] .= imagenes[i];
    end;
    return nuevoArray;
end;

train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

labels = unique(targets)
train_set = (train_imgs, onehotbatch(train_labels, labels))
test_set = (test_imgs, onehotbatch(test_labels, labels))
println(test_set[2]')


println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))

println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");

funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
ann1 = Chain(

    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #570, 760 -> 285, 380
    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #285, 380 -> 142, 190
    MaxPool((2,2)),

    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #142, 190 -> 71, 95
    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    #71*95*32
    Dense(215840, 3),

    softmax
)

ann = Chain(

    Conv((61, 81), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #570, 760 -> 285, 380
    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #285, 380 -> 142, 190
    MaxPool((2,2)),

    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    #142, 190 -> 71, 95
    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    #71*95*32
    Dense(215840, 3),

    softmax
)



# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
loss(ann1, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann1(x),y) : Losses.crossentropy(ann1(x),y);
eta = 0.01;
opt_state = Flux.setup(Adam(eta), ann1);

println("salidas deseadas:")
println(test_set[2])
println("size salidas ann:")
println(size((ann1(test_set[1]))))
println("Salidas ann")
println((ann1(test_set[1])))
println(onecold(ann1(test_set[1])))

println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;


while !criterioFin

    # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

    # Se entrena un ciclo
    Flux.train!(loss, ann1, [train_set], opt_state);

    numCiclo += 1;

    # Se calcula la precision en el conjunto de entrenamiento:
    #precisionEntrenamiento = mean(onecold(ann1(train_set[1])) .== onecold(train_set[2]));
    println("hola")
    precisionEntrenamiento = accuracy(ann1(train_set[1])', train_set[2]')
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
    if (precisionEntrenamiento > mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        println(onecold(ann1(test_set[1])))
        println(test_set[2])
        println(onecold(test_set[2]))
        precisionTest = mean(onecold(ann1(test_set[1])) .== onecold(test_set[2]));
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(ann1);
        numCicloUltimaMejora = numCiclo;
    end

    # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
    if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
        global eta
        eta /= 10.0
        println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
        adjust!(opt_state, eta)
        numCicloUltimaMejora = numCiclo;
    end

    # Criterios de parada:

    # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
    if (precisionEntrenamiento >= 0.999)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true;
    end

    # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end
end
