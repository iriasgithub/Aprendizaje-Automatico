using DataFrames

#Implementación de la función confusionMatrix para problemas de dos clases
#outputs: vector con las salidas obtenidas para un modelo
#targets: salidas deseadas para un modelo 
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((outputs .== 0) .& (targets .== 0))
    FP = sum((outputs .== 1) .& (targets .== 0))
    FN = sum((outputs .== 0) .& (targets .== 1))
    if VP == FN == 0
        recall = 0.0
    else recall = VP/(FN + VP)
    end 
    if VP == FP == 0
        precision = 1.0
    else precision = VP/(VP + FP)
    end
    if VN == FP == 0
        specificity = 0.0
    else specificity = VN/(FP + VN)
    end
    if VN == FN == 0
        NPV = 0.0
    else NPV = VN/(VN + FN)
    end
    if recall == precision == 0.0
        f1_score = 0.0
    else f1_score = 2*(recall * precision)/(recall + precision)
    end
    accuracy = (VN + VP)/(VN + VP + FN + FP)
    error_rate = (FN + FP)/(VN + VP + FN + FP) 
    confusion_matrix = [VN FP; FN VP]

    return(accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, confusion_matrix)
end 

function confusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    confusionMatrix(outputs, targets)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1}) 

    (accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, matriz_confusion) = confusionMatrix(outputs, targets)

    df_confusion = DataFrame(Negativo = matriz_confusion[1,:], Positivo = matriz_confusion[2,:])

    println("Accuracy = " * string(accuracy))
    println("Error rate = " * string(error_rate))
    println("Recall = " * string(recall))
    println("Specificity = " * string(specificity))
    println("Precision = " * string(precision))
    println("NPV = " * string(NPV))
    println("f1_score = " * string(f1_score))
    println(df_confusion)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    (accuracy, error_rate, recall, specificity, precision, 
    NPV, f1_score, matriz_confusion) = confusionMatrix(outputs, targets; threshold)
    
    df_confusion = DataFrame(Negativo = matriz_confusion[1,:], Positivo = matriz_confusion[2,:])

    println("Accuracy = " * string(accuracy))
    println("Error rate = " * string(error_rate))
    println("Recall = " * string(recall))
    println("Specificity = " * string(specificity))
    println("Precision = " * string(precision))
    println("NPV = " * string(NPV))
    println("f1_score = " * string(f1_score))
    println(df_confusion)
end

#a1 = [true, true, false, true, false]
#a2 = [true, false, true, false, false]
#printConfusionMatrix(a1, a2)