using Random 
#n: número de patrones 
#p: valor entre 0 y 1 que indica el porcentaje de patrones que se separarán para el conjunto de test
function holdOut(N::Int , P::Real)
    length_test = round(Int, N * P)
    println(length_test)
    v = randperm(N)
    v_test = v[1:length_test]
    v_train = v[(length_test + 1):N]
    @assert ((length(v_test) + length(v_train)) == N) "La suma de la longitud del vector de entrenamiento
     y el vector de test debe ser igual al numero total de patrones"
    return (v_train, v_test)
end

#n: número de patrones 
#pval: tasa de patrones en el conjunto de validación
#ptest: tasa de patrones en el conjunto de test
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert (Pval + Ptest < 1) "La suma de las dos tasas de patrones debe ser menor que 1"
    #Nos quedamos con los patrones de entrenamiento
    (v_train, index_rest) = holdOut(N, Ptest + Pval)
    #Nos quedamos con los indices de los patrones de test y validación
    (index_test, index_val) = holdOut(length(index_rest), Pval / (Pval + Ptest))
    #Tomamos esos índices del contenido del vector de rest
    v_test = index_rest[index_test]
    v_val = index_rest[index_val]
    @assert ((length(v_test) + length(v_train) + length(v_val)) == N) "La suma de la longitud del vector de entrenamiento
    y el vector de test debe ser igual al numero total de patrones"
    return (v_train, v_val, v_test)
end

#dataset contiene un patrón en cada fila. Necesario trasponer las matrices para 
#entrenar un ciclo, calcular valor de loss o tomar matrices de salidas que se 
#obtienen al ejecutar la RNA.
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 
    
    inputs = trainingDataset[1]'
    targets = trainingDataset[2]'
    v_train_loss = []
    v_val_loss = []
    v_test_loss = []
    epochs_without_improvement = 0
    numInputs = size(inputs, 1)
    numOutputs = size(targets, 1)
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    opt_state = Flux.setup(Adam(learningRate), ann)

    best_rna = deepcopy(ann);

    epoch = 1
    while epoch <= maxEpochs && epochs_without_improvement < maxEpochsVal && loss(ann, inputs, targets) > minLoss

        if !isequal(validationDataset, ((Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))))
            val_loss_best_ann= loss(best_rna, validationDataset[1]', validationDataset[2]')
        end
        Flux.train!(loss, ann, [(inputs,targets)], opt_state)

        if !isequal(validationDataset, ((Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))))
            val_loss = loss(ann, validationDataset[1]', validationDataset[2]')
            if val_loss < val_loss_best_ann
                best_rna = deepcopy(ann);
                epochs_without_improvement = 0
            else 
                epochs_without_improvement += 1
            end
            push!(v_val_loss, val_loss)
            push!(v_test_loss, loss(ann, testDataset[1]', testDataset[2]'))
        end 
        best_rna = deepcopy(ann)
        push!(v_train_loss, loss(ann, inputs, targets))

        epoch += 1
        
    end 
    return (best_rna, v_train_loss, v_val_loss, v_test_loss)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 

    trainingSet = (trainingDataset[1], reshape(trainingDataset[2], (size(trainingDataset[2], 2), 1)))
    validationSet = (validationDataset[1], reshape(validationDataset[2], (size(validationDataset[2], 2), 1)))
    testSet = (testDataset[1], reshape(testDataset[2], (size(testDataset[2], 2), 1)))
    trainClassANN(topology, trainingSet; validationSet, testSet, transferFunctions = transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal = maxEpochsVal)
end