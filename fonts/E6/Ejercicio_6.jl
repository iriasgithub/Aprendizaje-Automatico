#Función para obtener el separador del sistema
function separador_dir()
    if Sys.iswindows()
        return "\\"
    else
        return "/"
    end
end

sep = separador_dir()

ruta = pwd()
ejercicio_4_1 = ruta*"$sep"*"fonts"*"$sep"*"E4"*"$sep"*"ejercicio_4.1.jl"
ejercicio_4_2 = ruta*"$sep"*"fonts"*"$sep"*"E4"*"$sep"*"ejercicio_4.2.jl"
ejercicio_5 = ruta*"$sep"*"fonts"*"$sep"*"E5"*"$sep"*"ejercicio_5.jl"
include(ejercicio_4_1)
include(ejercicio_4_2)
include(ejercicio_5)

using ScikitLearn 

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier 
@sk_import neighbors: KNeighborsClassifier

#modelType: indicador del modelo a entrenar. (:ANN, :SVC, :DecisionTreeClassifier y :KNeighborsClassifier).
#modelHyperparameters: hiperparámetros del modelo (pueden faltar algunos porque en :ANN no son todos obligatorios)
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})
    if modelType == :ANN
        #Creación de un diccionario para pasar los valores
        args = Dict()
        #Asignación de los campos obligatorios
        args["topology"] = modelHyperparameters["topology"]
        args["inputs"] = inputs
        args["targets"] = targets   
        args["crossValidationIndices"] = crossValidationIndices
        #Asignación de los campos opcionales
        #numExecutions = haskey(modelHyperparameters, "numExecutions") ? args["numExecutions"] = modelHyperparameters["numExecutions"] : args["numExecutions"] = 50
        #transferFunctions = haskey(modelHyperparameters, "transferFunctions") ? args["transferFunctions"] = modelHyperparameters["transferFunctions"] : args["transferFunctions"] = fill(σ, length(topology))
        #maxEpochs = haskey(modelHyperparameters, "maxEpochs") ? args["maxEpochs"] =  modelHyperparameters["maxEpochs"] : args["maxEpochs"] = 1000
        #minLoss = haskey(modelHyperparameters, "minLoss") ? args["minLoss"] = modelHyperparameters["minLoss"] : args["minLoss"] = 0.0
        #learningRate = haskey(modelHyperparameters, "learningRate") ? args["learningRate"] = modelHyperparameters["learningRate"] : args["learningRate"] = 0.01
        #validationRatio = haskey(modelHyperparameters, "validationRatio") ? args["validationRatio"] = modelHyperparameters["validationRatio"] : args["validationRatio"] = 0
        #maxEpochsVal = haskey(modelHyperparameters, "maxEpochsVal") ? args["maxEpochsVal"] = modelHyperparameters["maxEpochsVal"] : args["maxEpochsVal"] = 20

        return ANNCrossValidation(args["topology"], args["inputs"], args["targets"], args["crossValidationIndices"])

    else
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
        #Transformación del vector de salidas deseadas a un vector de cadenas de texto
        targets = string.(targets); 
        #Creación del modelo correspondiente
        if modelType == :SVC 
            model = SVC(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
                    degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
                    coef0=modelHyperparameters["coef0"]); 
        elseif modelType == :DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"])
        elseif modelType == :KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"])
        end 
        #Se comienza con el bucle de validación cruzada
        for fold in 1:folds
            #Creacion de las variables de entrada y salida deseada para entrenamiento y test a partir de los índices.
            train_indices = findall(x -> x != fold, crossValidationIndices)
            test_indices = findall(x -> x == fold, crossValidationIndices)
            
            train_inputs = inputs[train_indices, :]
            train_targets = targets[train_indices]
            test_inputs = inputs[test_indices, :]
            test_targets = targets[test_indices]
            #Entrenar el modelo (solo una vez porque son modelos determinísticos)

            fit!(model, train_inputs, train_targets)
            #Aplicar conjunto de test
            test_outputs = predict(model, test_inputs)
            test_outputs = cadena_a_bool.(test_outputs)
            test_targets = cadena_a_bool.(test_targets)
            #Calcular métricas
            (v_accuracy[fold], v_error_rate[fold], v_recall[fold],
            v_specificity[fold], v_precision[fold], v_NPV[fold], v_f1_score[fold], _) = 
            confusionMatrix(vec(test_outputs), vec(test_targets))
            
        end
        return((mean(v_accuracy), std(v_accuracy)),(mean(v_error_rate), std(v_error_rate)),(mean(v_recall), std(v_recall)),
        (mean(v_specificity), std(v_specificity)),(mean(v_precision), std(v_precision)),(mean(v_NPV), std(v_NPV)),(mean(v_f1_score), std(v_f1_score)))
    end
end

#Funciones para normalización
function cadena_a_bool(cadena::String)
    if cadena == "true"
        return true
    elseif cadena == "false"
        return false
    else
        error("La cadena debe ser 'true' o 'false'")
    end
end
