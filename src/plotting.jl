function separador_dir()
    if Sys.iswindows()
        return "\\"
    else
        return "/"
    end
end

sep = separador_dir()
ruta = pwd()

ejercicio_6 = ruta*"$sep"*"fonts"*"$sep"*"E6"*"$sep"*"ejercicio_6.jl"
include(ejercicio_6)

function convertir_a_strings(vector::Vector{Any})
    vector_de_strings = []
    for arreglo in vector
        push!(vector_de_strings, "[" * join(string.(arreglo), ", ") * "]")
    end
    return vector_de_strings
end

function execute_ANN(topologias::Vector{Vector{Int}}, inputs::AbstractArray{<:Real, 2}, targets::AbstractArray{<:Any, T}) where T
    mean_f1_plot = []
    std_f1_plot = []
    topo_plot = []
    for topologia in topologias
        push!(topo_plot, topologia)
        args = Dict("topology" => topologia)
        println("Entrenando y evaluando con topología: ", topologia)
        
        ((_,_), (_,_), (_,_), (_,_), (_,_), (_,_), (mean_f1_score, std_f1_score)) = 
        modelCrossValidation(:ANN, args, inputs, vec(targets), crossvalidation(targets, 10))
        
        println("Fin de entreno con topología: ", topologia)
        
        # Guardar el F1 Score asociado a la topología actual
        push!(mean_f1_plot, mean_f1_score)
        push!(std_f1_plot, std_f1_score)
    end
    
    # Generar gráfica de F1 Scores en función de la topología
    
    plot1 = plot(convertir_a_strings(topo_plot), mean_f1_plot, xlabel="Topología", ylabel="Media F1 Score", title="Media F1 Score vs. Topología", legend=false)
    display(plot1)
    plot2 = plot(convertir_a_strings(topo_plot), std_f1_plot, xlabel="Topología", ylabel="Desviación típica F1 Score", title="Desviación típica F1 Score vs. Topología", legend=false) 
    display(plot2)
    max_f1_mean = maximum(mean_f1_plot)
    println(std_f1_plot)
    return (max_f1_mean, topo_plot[findall(x -> x == max_f1_mean, mean_f1_plot)])
end

function execute_SVM(configurations::Matrix{Tuple{Float64, String}}, inputs::AbstractArray{<:Real, 2}, targets::AbstractArray{<:Any, T}) where T
    mean_f1_plot = []
    std_f1_plot = []
    cartesian_plot = []


    for config in configurations
        C_value, kernel_type = config
        push!(cartesian_plot, config)
        args = Dict("C" => C_value, "kernel" => kernel_type, "degree" => 3, "gamma" => "scale", "coef0" => 0.0)
        println("Entrenar y evaluar con el valor de C : ", C_value, " y el tipo kernel: ", kernel_type)

        # Entrenar y evaluar el modelo SVM con la configuración actual
        ((_,_), (_,_), (_,_), (_,_), (_,_), (_,_), (mean_f1_score, std_f1_score)) = 
        modelCrossValidation(:SVC, args, inputs, vec(targets), crossvalidation(targets, 10))
        
        println("Fin de entrenar y evaluar con el valor de C: ", C_value, " y el tipo kernel: ", kernel_type)

        # Guardar el puntaje F1 asociado con la configuración actual
        push!(mean_f1_plot, mean_f1_score)
        push!(std_f1_plot, std_f1_score)
    end

    # Generar gráfica de barras de puntajes F1 en función de las configuraciones
    plot1 = bar(convertir_a_strings(cartesian_plot), mean_f1_plot, size = (900, 500), xlabel="Configuration", ylabel="Media F1 Score", title="Media F1 Score vs. Configuration", legend=false)
    display(plot1)
    plot2 = bar(convertir_a_strings(cartesian_plot), std_f1_plot, size = (900, 500), xlabel="Configuration", ylabel="Desviación típica F1 Score", title="Desviación típica F1 Score vs. Configuration", legend=false)
    display(plot2)

    max_f1_mean = maximum(mean_f1_plot)
    println(std_f1_plot)
    return  (max_f1_mean, cartesian_plot[findall(x -> x == max_f1_mean, mean_f1_plot)])
end

function execute_ARB(profundidades::Vector{Int64}, inputs::AbstractArray{<:Real, 2}, targets::AbstractArray{<:Any, T}) where T    
    mean_f1_plot = []
    std_f1_plot = []
    prof_plot = []


    for prof in profundidades
        push!(prof_plot, prof)
        args = Dict("max_depth" => prof)
        println("Entrenar y evaluar con el valor de max_depth : ", prof)

        # Entrenar y evaluar el modelo con la configuración actual
        ((_,_), (_,_), (_,_), (_,_), (_,_), (_,_), (mean_f1_score, std_f1_score)) = 
        modelCrossValidation(:DecisionTreeClassifier, args, inputs, vec(targets), crossvalidation(targets, 10))
        
        println("Fin de entrenar y evaluar con el valor de max_depth : ", prof)

        # Guardar el puntaje F1 asociado con la configuración actual
        push!(mean_f1_plot, mean_f1_score)
        push!(std_f1_plot, std_f1_score)

    end

    # Generar gráfica de barras de puntajes F1 en función de las configuraciones

    plot1 = plot(prof_plot, mean_f1_plot, xlabel="Max_depth", ylabel="F1 Score", title="F1 Score vs. Max_depth", legend=false)
    display(plot1)
    plot2 = plot(prof_plot, std_f1_plot, xlabel="Max_depth", ylabel="Desviación típica F1 Score", title="Desviación típica F1 Score vs. Max_depth", legend=false)
    display(plot2)

    max_f1_mean = maximum(mean_f1_plot)
    println(std_f1_plot)
    return (max_f1_mean, prof_plot[findall(x -> x == max_f1_mean, mean_f1_plot)])
end

function execute_KNN(neighbors::Vector{Int64}, inputs::AbstractArray{<:Real, 2}, targets::AbstractArray{<:Any, T}) where T
    mean_f1_plot = []
    std_f1_plot = []
    neighbor_plot = []


    for neighbor in neighbors
        push!(neighbor_plot, neighbor)
        args = Dict("n_neighbors" => neighbor)
        println("Entrenar y evaluar con el valor de n_neighbors : ", neighbor)

        # Entrenar y evaluar el modelo con la configuración actual
        ((_,_), (_,_), (_,_), (_,_), (_,_), (_,_), (mean_f1_score, std_f1_score)) = 
        modelCrossValidation(:KNeighborsClassifier, args, inputs, vec(targets), crossvalidation(targets, 10))
        
        println("Fin de entrenar y evaluar con el valor de n_neighbors : ", neighbor)

        # Guardar el puntaje F1 asociado con la configuración actual
        push!(mean_f1_plot, mean_f1_score)
        push!(std_f1_plot, std_f1_score)
    end

    # Generar gráfica de barras de puntajes F1 en función de las configuraciones

    plot1 =  plot(neighbor_plot, mean_f1_plot, xlabel="Nearest Neighbors", ylabel="F1 Score", title="F1 Score vs. Nearest Neighbors", legend=false)
    display(plot1)
    plot2 = plot(neighbor_plot, std_f1_plot, xlabel="Nearest Neighbors", ylabel="Desviación típica F1 Score", title="Desviación típica F1 Score vs. Nearest Neighbors", legend=false)
    display(plot2)

    max_f1_mean = maximum(mean_f1_plot)
    
    println(std_f1_plot)
    return (max_f1_mean, neighbor_plot[findall(x -> x == max_f1_mean, mean_f1_plot)])
end