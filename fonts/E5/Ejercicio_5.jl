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
ejercicio_4_1 = ruta*"$sep"*"fonts"*"$sep"*"E4"*"$sep"*"ejercicio_4.1.jl"
ejercicio_4_2 = ruta*"$sep"*"fonts"*"$sep"*"E4"*"$sep"*"ejercicio_4.2.jl"
include(ejercicio_2)
include(ejercicio_3)
include(ejercicio_4_1)
include(ejercicio_4_2)

#Se realizan k experimentos. 
#En cada experimento se dividen los patrones en k subconjuntos disjuntos donde el k subconjunto se deja para test y los k-1 para train.
#El valor de test final será una media de los valores de test de todos los experimentos.
#Se garantiza que todos los datos se usan al menos una vez en test y train. (evaluación ligeramente pesimista)

using Random

#Fijamos semilla aleatoria
Random.seed!(1234)

#N: número de patrones
#k: número de subconjuntos en los que se va a partir el conjunto de datos
function crossvalidation(N::Int64, k::Int64)
    v = collect(1:k)
    v = repeat(v, ceil(Int, N/k))
    v = v[1:N]
    shuffle!(v)
    return v
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64) 
    N = length(targets)
    @assert (N >= k) "Cada clase debe tener al menos k patrones"
    v = collect(1:N)
    positive_in_class = sum(targets)
    negative_in_class = N - positive_in_class
    @assert (positive_in_class >= k && negative_in_class >= k) "Cada clase debe tener al menos k patrones"
    v[targets] .= crossvalidation(positive_in_class, k)
    v[targets .== 0] .= crossvalidation(negative_in_class, k)
    return v
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N, classes = size(targets)
    v = collect(1:N)
    for class in 1:classes 
        positive_in_class = sum(targets[:, class]) 
        @assert (positive_in_class >= k) "Cada clase debe tener al menos k patrones"
        boolean_vector = targets[:, class]
        index = findall(boolean_vector)
        v[boolean_vector] .= crossvalidation(positive_in_class, k)
    end
    return v
end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets)
    targets = oneHotEncoding(targets, classes)
    crossvalidation(targets, k)
    #Desarrollarla sin oneHotEncoding?
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1},
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20) 

    #Cálculo del n de folds que se desea hacer (se hacen k experimentos y k subconjuntos)
    folds = maximum(crossValidationIndices)
    #Vectores para almacenar el resultado del entrenamiento de la RNA de cada métrica
    v_accuracy = zeros(folds)
    v_error_rate = zeros(folds)
    v_recall = zeros(folds) 
    v_specificity = zeros(folds) 
    v_precision = zeros(folds) 
    v_NPV = zeros(folds) 
    v_f1_score = zeros(folds)
    v_confusion_matrices = []

    classes = length(unique(targets))
    targets = oneHotEncoding(targets)

    
    for fold in 1:folds
        #Creacion de las variables de entrada y salida deseada para entrenamiento y test a partir de los índices.
        train_indices = findall(x -> x != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)
        
        train_inputs = inputs[train_indices, :]
        train_targets = targets[train_indices, :]
        test_inputs = inputs[test_indices, :]
        test_targets = targets[test_indices, :]
        
        #CÁLCULO DE LOS PARÁMETROS DE NORMALIZACIÓN EN BASE ÚNICAMENE A LOS TRAIN_INPUTS DE ESTE FOLD:
        normalizationParameters = calculateMinMaxNormalizationParameters(train_inputs)
        normalizeMinMax!(train_inputs, normalizationParameters)
        normalizeMinMax!(test_inputs, normalizationParameters)

        #Crear los vectores para guardar las métricas en cada entrenamiento
        v_train_accuracy = zeros(numExecutions)
        v_train_error_rate = zeros(numExecutions)
        v_train_recall = zeros(numExecutions) 
        v_train_specificity = zeros(numExecutions) 
        v_train_precision = zeros(numExecutions) 
        v_train_NPV = zeros(numExecutions) 
        v_train_f1_score = zeros(numExecutions)

        for execution in 1:numExecutions
            testDataset = (test_inputs, test_targets)
            if (validationRatio > 0.0)
                #Dividimos el conjunto de entrenamiento entre entrenamiento y validación
                #Calculamos el número de patrones, del conjunto de entrenamiento inicial que queremos dividir (será el numero de filas de cualquiera de las
                #dos matrices, la de train_targets o train_inputs)
                n_instances = size(train_targets, 1)
                #Calculamos el ratio de patrones para validación. El validationRatio es el ratio para el conjunto total de datos (hacemos regla de 3)
                Pval = n_instances * validationRatio / size(targets, 1)
                #train_index y val_index contienen los indices de los patrones de train y validación en train_inputs y train_targets
                (train_index, val_index) = holdOut(n_instances, Pval)
                #Dividimos el conjunto de train en train y validación
                trainingDataset = (train_inputs[train_index, :], train_targets[train_index, :])
                validationDataset = (train_inputs[val_index, :], train_targets[val_index, :])
                (ann, _, _, _) = trainClassANN(topology, trainingDataset; transferFunctions, maxEpochs, minLoss, learningRate, validationDataset, testDataset, maxEpochsVal)
                test_outputs = ann(test_inputs')'
                
            else
                trainingDataset = (train_inputs, train_targets)
                (ann, _) = trainClassANN(topology, trainingDataset; transferFunctions, maxEpochs, minLoss, learningRate)
                test_outputs = ann(test_inputs')'
            end

            if(classes > 2)
                (v_train_accuracy[execution], v_train_error_rate[execution], v_train_recall[execution],
                v_train_specificity[execution], v_train_precision[execution], v_train_NPV[execution], v_train_f1_score[execution], matriz_confusion) = 
                confusionMatrix(test_outputs, test_targets)
            else
                (v_train_accuracy[execution], v_train_error_rate[execution], v_train_recall[execution],
                v_train_specificity[execution], v_train_precision[execution], v_train_NPV[execution], v_train_f1_score[execution], matriz_confusion) = 
                confusionMatrix(vec(test_outputs), vec(test_targets))
            end
            push!(v_confusion_matrices, matriz_confusion)
        end

        v_accuracy[fold] = mean(v_train_accuracy)
        v_error_rate[fold] = mean(v_train_error_rate)
        v_recall[fold] = mean(v_train_recall)
        v_specificity[fold] = mean(v_train_specificity)
        v_precision[fold] = mean(v_train_precision)
        v_NPV[fold] = mean(v_train_NPV)
        v_f1_score[fold] = mean(v_train_f1_score)
    end

    avg_confusion_matrix = (sum(v_confusion_matrices) / folds) / numExecutions
    println(avg_confusion_matrix)

    return((mean(v_accuracy), std(v_accuracy)),(mean(v_error_rate), std(v_error_rate)),(mean(v_recall), std(v_recall)),
    (mean(v_specificity), std(v_specificity)),(mean(v_precision), std(v_precision)),(mean(v_NPV), std(v_NPV)),(mean(v_f1_score), std(v_f1_score)))
end


