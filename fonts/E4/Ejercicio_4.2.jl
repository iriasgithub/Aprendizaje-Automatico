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

include("Ejercicio_4.1.jl")

#outputs: Salidas del modelo booleanas con un patron en cada fila y clase en cada columna.
#targets: Salidas deseadas booleanas con un patron en cada fila y clase en cada columna.
#weighted: booleano que indica weighted o macro.
function confusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    classes = size(outputs,2)
    @assert (classes == size(targets,2) & classes != 2) 
        "Las matrices de salidas deseadas y salidas emitidas deben tener el mismo número de columnas y debe ser distinto de 2"
    if classes == 1
       outputs = vec(outputs) 
       targets = vec(targets)
       (accuracy, error_rate, recall, specificity, precision, 
        NPV, f1_score, confusion_matrix) = confusionMatrix(outputs, targets)
    end
    v_recall = zeros(classes) 
    v_specificity = zeros(classes) 
    v_precision = zeros(classes) 
    v_NPV = zeros(classes) 
    v_f1_score = zeros(classes)
    for class in 1:classes
        v_outputs = vec(outputs[:, class])
        v_targets = vec(targets[:, class])
        (_, _, v_recall[class], v_specificity[class], v_precision[class], 
        v_NPV[class], v_f1_score[class], _) = confusionMatrix(v_outputs, v_targets)
    end
    confusion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:classes, j in 1:classes]
    if weighted
        weights = [sum(targets[:, i]) for i in 1:classes]
        weights_value = sum(weights)
        recall = sum(weights .* v_recall) / weights_value
        specificity = sum(weights .* v_specificity) / weights_value
        precision = sum(weights .* v_precision) / weights_value
        NPV = sum(weights .* v_NPV) / weights_value
        f1_score = sum(weights .* v_f1_score) / weights_value
    else
        recall = sum(v_recall) / classes
        specificity = sum(v_specificity) / classes
        precision = sum(v_precision) / classes
        NPV = sum(v_NPV) / classes
        f1_score = sum(v_f1_score) / classes
    end

    classComparison = targets .== outputs 
    correctClassifications = all(classComparison, dims=2)
    accuracy = mean(correctClassifications)
    error_rate = 1 - accuracy

    return(accuracy, error_rate, recall, specificity, precision, NPV, f1_score, confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs = classifyOutputs(outputs)
    confusionMatrix(outputs, targets, weighted = weighted)
end 

function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #Línea de programación defensiva que permite asegurarse de que todas las clases del vector de
    #salidas están incluidas en el vector de salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs])) 
    output_classes = unique(outputs)
    target_classes = unique(targets)
    outputs = oneHotEncoding(outputs, output_classes)
    targets = oneHotEncoding(targets, target_classes)
    confusionMatrix(outputs, targets, weighted = weighted)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    (accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, matriz_confusion) = confusionMatrix(outputs, targets, weighted = weighted)

    df_confusion = DataFrame(matriz_confusion, :auto)

    println("Accuracy = " * string(accuracy))
    println("Error rate = " * string(error_rate))
    println("Recall = " * string(recall))
    println("Specificity = " * string(specificity))
    println("Precision = " * string(precision))
    println("NPV = " * string(NPV))
    println("f1_score = " * string(f1_score))
    println(df_confusion)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    (accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, matriz_confusion) = confusionMatrix(outputs, targets, weighted = weighted)

    df_confusion = DataFrame(matriz_confusion, :auto)

    println("Accuracy = " * string(accuracy))
    println("Error rate = " * string(error_rate))
    println("Recall = " * string(recall))
    println("Specificity = " * string(specificity))
    println("Precision = " * string(precision))
    println("NPV = " * string(NPV))
    println("f1_score = " * string(f1_score))
    println(df_confusion)

end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)

    (accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, matriz_confusion) = confusionMatrix(outputs, targets, weighted = weighted)

    df_confusion = DataFrame(matriz_confusion, :auto)

    println("Accuracy = " * string(accuracy))
    println("Error rate = " * string(error_rate))
    println("Recall = " * string(recall))
    println("Specificity = " * string(specificity))
    println("Precision = " * string(precision))
    println("NPV = " * string(NPV))
    println("f1_score = " * string(f1_score))
    println(df_confusion)
end

#outputs = [0 1 0; 1 0 0; 1 0 0; 0 0 1]
#outputs = Bool.(outputs)
#targets = [1 0 0; 1 0 0; 0 1 0; 0 0 1]
#targets = Bool.(targets)

#println(targets)
#println(outputs)

#printConfusionMatrix(outputs, targets, weighted=false)