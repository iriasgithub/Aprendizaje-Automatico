using Statistics

#feature: valores categóricos del atributo o salida deseada para cada patrón.
#classes: valores de las categorías.

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    n_classes = length(classes)
    if n_classes == 2
        result = feature .== classes[1]
        result = reshape(result, (length(result), 1))
    elseif n_classes > 2
       result = Array{Bool, 2}(undef, length(feature), n_classes)
       for column in 1:n_classes
        result[:, column] = feature .== classes[column]
       end
    end
    return result
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    classes = unique(feature)
    result = feature .== classes[1]
    result = reshape(result, (length(result), 1))
    return result
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims = 1))
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (mean(dataset, dims=1), std(dataset, dims=1))
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    n_atributes = size(dataset,2)
    #Recorremos los atributos de dataset
    #Para cada atributo  
    for atribute in 1:n_atributes
        #si el máximo es igual al mínimo, le asignamos un valor de 0 a toda la columna
        if normalizationParameters[2][atribute] == normalizationParameters[1][atribute]
            dataset[:, atribute] .= 0
        #sino aplicamos la normalización 
        else 
            dataset[:, atribute] = (dataset[:, atribute] .- normalizationParameters[1][atribute]) ./
             (normalizationParameters[2][atribute] .- normalizationParameters[1][atribute])
        end
    end
    return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied_dataset = copy(dataset)
    normalizeMinMax!(copied_dataset, normalizationParameters)
    return copied_dataset
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}) 
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax(dataset, normalizationParameters)
end

#Opción normalizeMinMax! sin bucles
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    # Aplicar valores de 0 si el máximo es igual al mínimo
    #Se crea una matriz de booleanos con 1 donde el máximo es igual al mínimo usando .==
    #Se aplica la función ifelse que sustituye los elementos que deben ser reemplazados (indicados por
    #la matriz de booleanos) de la matriz dataset por 0
    dataset .= ifelse.(normalizationParameters[2] .== normalizationParameters[1], 0, dataset)
    # Realizar la normalización 
    dataset .= (dataset .- normalizationParameters[1]) ./ (normalizationParameters[2] .- normalizationParameters[1])
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    n_atributes = size(dataset,2)
    #Recorremos los atributos de inputs
    #Para cada atributo  
    for atribute in 1:n_atributes
        #si la desviación típica es igual a 0, le asignamos un valor de 0 a toda la columna
        if normalizationParameters[2][atribute] == 0
            dataset[:, atribute] .= 0
        #sino aplicamos la normalización 
        else 
            dataset[:, atribute] =  (dataset[:, atribute] .- normalizationParameters[1][atribute]) ./ normalizationParameters[2][atribute]
        end
    end
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) 
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters) 
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied_dataset = copy(dataset)
    normalizeZeroMean!(copied_dataset, normalizationParameters)
    return copied_dataset
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean(dataset, normalizationParameters)
end

#Opción normalizeZeroMean! sin bucles
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    dataset .= ifelse.(normalizationParameters[2] .== 0, 0, dataset)
    dataset .= (dataset .- normalizationParameters[1]) ./ normalizationParameters[2]
end

#outputs: vector con las salidas de un modelo que emite una única salida para un problema de clasificación binaria
#threshold: umbral que define la pertenencia a una clase u otra.
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5) 
    return outputs .>= threshold
end

#outputs: matriz con las salidas y un patrón en cada fila.
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    n_rows, n_classes = size(outputs)
    if n_classes == 1
        boolean_vector = classifyOutputs(outputs[:]; threshold = threshold)
        boolean_matrix = reshape(boolean_vector, (n_rows, 1))
    elseif n_classes > 1
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        boolean_matrix = falses(size(outputs))
        boolean_matrix[indicesMaxEachInstance] .= true
    end
    return boolean_matrix
end 

#targets: salidas deseadas con un patrón en cada fila
#outputs: salidas emitidas para un modelo con un patrón en cada fila
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert (size(outputs,1) == size(targets,1)) "Las matrices de salidas deseadas y salidas emitidas deben tener el mismo número de filas"
    boolean_matrix = outputs .== targets
    return mean(boolean_matrix)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert (size(outputs,1) == size(targets,1)) "Las matrices de salidas deseadas y salidas emitidas deben tener el mismo número de filas"
    n_classes = size(outputs,2)
    if n_classes == 1
        accuracy = accuracy(outputs[:,1], targets[:,1])
    elseif n_classes > 2 
        classComparison = targets .== outputs 
        correctClassifications = all(classComparison, dims=2)
        accuracy = mean(correctClassifications)
    end
    return accuracy
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
    threshold::Real=0.5) 
    accuracy = accuracy(outputs .>= threshold, targets)
    return accuracy
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5) 
    if size(outputs, 2) == 1
        return accuracy(targets[:,1], outputs[:,1])
    else
        return accuracy(targets, classifyOutputs(outputs))
    end
end


using Flux
using Flux.Losses

#numInputs: número de neuronas de la capa de entrada
#numOutputs: número de neuronas de la capa de salida
#topology: número de capas ocultas y neuronas en cada una de ellas
#transferFunctions: funciones de activación en cada capa oculta

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann = Chain()
    numInputsLayer = numInputs
    if !isempty(topology)
        for (numOutputsLayer, activationFunction) = zip(topology, transferFunctions)
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, activationFunction))
            numInputsLayer = numOutputsLayer
        end
    end
    #Creación de la capa final
    #Problemas de clasificación con dos clases
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, sigmoid))
    #Problemas de clasificación con más de dos clases
    elseif numOutputs >= 3 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

#dataset contiene un patrón en cada fila. Necesario trasponer las matrices para 
#entrenar un ciclo, calcular valor de loss o tomar matrices de salidas que se 
#obtienen al ejecutar la RNA.
function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    inputs = dataset[1]'
    targets = dataset[2]'
    loss_vector = zeros(maxEpochs)
    numInputs = size(inputs, 1)
    numOutputs = size(targets, 1)
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(learningRate), ann)
    for epoch in 1:maxEpochs
      
        Flux.train!(loss, ann, [(inputs,targets)], opt_state)
        loss_vector[epoch] = loss(ann, inputs, targets)
        if loss(ann, inputs, targets) <= minLoss
            break
        end
    end 
    return (ann, loss_vector)
end 

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    targets = reshape(targets, (size(targets, 1), 1))
    trainClassANN(topology, (inputs, targets); transferFunctions = transferFunctions, maxEpochs, minLoss, learningRate)
end