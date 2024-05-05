using Images
using ImageFeatures

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
ejercicio_5 = ruta*"$sep"*"fonts"*"$sep"*"E5"*"$sep"*"ejercicio_5.jl"
ejercicio_6 = ruta*"$sep"*"fonts"*"$sep"*"E6"*"$sep"*"ejercicio_6.jl"

include(ejercicio_2)
include(ejercicio_3)
include(ejercicio_4_1)
include(ejercicio_4_2)
include(ejercicio_5)
include(ejercicio_6)

#Función para convertir una imagen RGB a Gray (escala de grises)
imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));

#clase: melanoma, no_melanoma, atypical_nevus
#formato: lesion, dermoscopic
function cargar_imagenes(clase::String, formato::String, toGray::Bool)
    directorio_raiz = ruta*"$sep"*"datasets"*"$sep"*"$clase"*"$sep"
    images = []

    directorio = joinpath(directorio_raiz, formato)
    archivos = readdir(directorio)
    for archivo in archivos
        ruta_imagen = joinpath(directorio, archivo)
        imagen = load(ruta_imagen)
        push!(images, imagen)
    end
    if(toGray)
        return imageToGrayArray.(images)
    else
        return images
    end
end

function crear_inputs_targets_binarias(clase1::Vector{Matrix{Float64}}, clase2::Vector{Matrix{Float64}}, index::Vector{Int64})

    clases = vcat(clase1, clase2)
    targets = vcat(falses(length(clase1)), trues(length(clase2)))
    clases = clases[index]
    targets = targets[index]

    return clases, targets
end

function crear_inputs_targets_binarias(clase1::Vector{Any}, clase2::Vector{Any}, index::Vector{Int64})

    clases = vcat(clase1, clase2)
    targets = vcat(falses(length(clase1)), trues(length(clase2)))
    clases = clases[index]
    targets = targets[index]

    return clases, targets
end

function crear_inputs_targets_multiclase(clase1::Vector{Any}, clase2::Vector{Any}, clase3::Vector{Any}, index::Vector{Int64})
    l1, l2, l3 = length(clase1), length(clase2), length(clase3)
    cod1 = fill("melanoma", l1)
    cod2 = fill("no_melanoma", l2)
    cod3 = fill("atypical_nevus", l3)
    inputs = vcat(cod1, cod2, cod3) 
    targets = inputs[index, :]
    inputs = vcat(clase1, clase2, clase3)[index]
   
    return inputs, vec(targets)
end

function crear_inputs_targets_multiclase(clase1::Vector{Matrix{Float64}}, clase2::Vector{Matrix{Float64}}, clase3::Vector{Matrix{Float64}}, index::Vector{Int64})
    l1, l2, l3 = length(clase1), length(clase2), length(clase3)
    cod1 = fill("melanoma", l1)
    cod2 = fill("no_melanoma", l2)
    cod3 = fill("atypical_nevus", l3)
    inputs = vcat(cod1, cod2, cod3)
    targets = oneHotEncoding(inputs, ["melanoma", "no_melanoma", "atypical_nevus"])
    inputs = vcat(clase1, clase2, clase3)[index]
    targets = targets[index, :]
    
    return inputs, targets
end

#Función para crear el bounding box de una imagen 
#image: imagen a crear el bounding box
#name: nombre con el que se guardará la imagen en la carpeta
function create_bounding_box(image::Matrix{Float64})

    # Se obtiene el boundingBox de la imagen
    labeled_array = ImageMorphology.label_components(Int.(image))

    # Se obtienen los centroides de los objetos
    centroides = ImageMorphology.component_centroids(labeled_array)[1:end];

    # Se obtienen las cajas de los objetos
    imagenRgb = RGB.(labeled_array, labeled_array, labeled_array);

    # Se pintan los centroides
    for centroide in centroides
        x = Int(round(centroide[1]));
        y = Int(round(centroide[2]));
        imagenRgb[ x, y ] = RGB(1,0,0);
    end;

    boundingBoxes = ImageMorphology.component_boxes(labeled_array)[1:end]

    indices = boundingBoxes[1]

    x1, y1 = Tuple(indices[1])  
    x2, y2 = Tuple(indices[end])  

    imagenRgb[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagenRgb[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagenRgb[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagenRgb[ x2 , y1:y2 ] .= RGB(0,1,0);

    return (x1, x2, y1, y2)

end 

#Primera característica: ratio de 1s respecto al total de toda la Bounding Box, esto lo que hace es medir la
#regularidad del lunar, lunares con mayor ratio tienden a
#image: imagen en escala de grises
#indexes: coordenadas de su bounding box 
function extract_regularity(image::Matrix{Float64}, indexes::NTuple{4, Int64})
    x1, x2, y1, y2 = indexes 
    bounded_image = image[x1:x2, y1:y2]
    ones_count = sum(bounded_image)
    total = (x2 - x1) * (y2 - y1)
    ratio = ones_count/total
    return ratio
end

#Segunda característica: Grado de simetría
#image: imagen en escala de grises
#indexes: coordenadas de su bounding box
function extract_asymmetry(image::Matrix{Float64}, indexes::NTuple{4, Int64})
    x1, x2, y1, y2 = indexes 
    bounded_image = image[x1:x2, y1:y2]
    num_columns = size(bounded_image, 2)
    if num_columns % 2 == 0
        median_index = div(num_columns, 2)
        part1 = bounded_image[:, 1:median_index]
        part2_reversed = reverse(bounded_image[:, median_index+1:end])
    else 
        median_index = div(num_columns, 2) + 1 
        part1 = bounded_image[:, 1:median_index]
        part2_reversed = reverse(bounded_image[:, median_index:end])    
    end
    asimmetry = sum(abs.(part1 .- part2_reversed))
    return asimmetry
end


# Función para calcular el área de un polígono
function calcular_area_poligono(convex_hull_points)
    area = 0
    n = length(convex_hull_points)
    for i in 1:n
        x1, y1 = convex_hull_points[i]
        x2, y2 = convex_hull_points[mod1(i+1, n)]
        area += (x1*y2 - x2*y1)
    end
    return abs(area)/2
end

# Mejora de la característica de irregularidad mediante convex hulls
function extract_irregularity(image::Matrix{Float64})
    
    convex_hull_points = convexhull(Bool.(image))
    convex_hull_points = [(index[1], index[2]) for index in convex_hull_points]
    
    # Calcular el área del convex hull 
    convex_hull_area = calcular_area_poligono(convex_hull_points)
    # Calcular el área de la máscara binaria
    binary_area = sum(image)

    irregularity = convex_hull_area / binary_area

    return irregularity
end

# Función para calcular la media del valor de los píxeles dentro del bounding box
function extract_mean_pixels_value(image::AbstractMatrix, binary_mask::AbstractMatrix)
    
    indices = findall(x -> x == 1.0, binary_mask)
    roi = image[indices]

    # Calcular la media del valor de los píxeles en la ROI
    media = mean(roi)

    return media
end

function calcular_longitud_borde(imagen::Matrix{Float64})
    m, n = size(imagen)
    border_length = 0

    # Recorrer los píxeles de la imagen
    for i in 1:m, j in 1:n
        # Verificar si el píxel actual es parte del borde
        if imagen[i, j] == 1
            # Verificar si algún vecino es diferente de 1
            if i == 1 || j == 1 || i == m || j == n || imagen[i-1, j] == 0 || imagen[i+1, j] == 0 || imagen[i, j-1] == 0 || imagen[i, j+1] == 0
                border_length += 1  # Incrementar la longitud del borde
            end
        end
    end

    return border_length
end

function extract_border_length(image::Matrix{Float64})
    
    longitud_borde = calcular_longitud_borde(image)
    n, m = size(image)
    longitud_borde_normalizada = longitud_borde / (n*m)

    return longitud_borde_normalizada
end

function aplicar_mascara(imagen::Matrix{RGB{N0f8}}, mascara::Array{Float64, 2}) #
    imagen_mascarada = copy(imagen)
    
    for i in 1:size(imagen, 1)
        for j in 1:size(imagen, 2)
            if mascara[i, j] == 1.0
                imagen_mascarada[i, j] = imagen[i, j]
            else
                imagen_mascarada[i, j] = RGB{N0f8}(0, 0, 0) 
            end
        end
    end
    
    return imagen_mascarada
end

function calcular_media_sin_ceros(matriz::Array{T, 2}) where T
    suma = zero(T)
    contador = 0
    for i in 1:size(matriz, 1)
        for j in 1:size(matriz, 2)
            for k in 1:size(matriz, 3)
                if matriz[i, j, k] != zero(T)
                    suma += matriz[i, j, k]
                    contador += 1
                end
            end
        end
    end

    if contador == 0
        return zero(T)
    else
        return suma / contador
    end
end

# Función para calcular la media del valor de los píxeles dentro del bounding box
function extract_mean_color_pixels_value(image::AbstractMatrix, binary_mask::AbstractMatrix)
    
    roi = aplicar_mascara(image, binary_mask)
    array_pixeles = float.(channelview(roi))
    array_pixeles_img = float.(channelview(image))
    red = array_pixeles[1, :, :]
    green = array_pixeles[2, :, :]
    blue = array_pixeles[3, :, :]

    media_red = calcular_media_sin_ceros(red)
    media_green = calcular_media_sin_ceros(green)
    media_blue = calcular_media_sin_ceros(blue)

    media = (media_red + media_blue + media_green)/ 3.0
    desv = std(array_pixeles_img)

    return media/desv
end



#Función para generar el dataset con las dos características
#inputs: vector de imágenes bmp
#axis: coordenadas de las bounding boxes de cada imagen
function extraer_caracteristicas_aprox1(inputs::Vector{Matrix{Float64}}, axis::Vector{Any}) 
    dataset = []
    for (image, ax) in zip(inputs, axis)
        attribute_1 = extract_regularity(image, ax)
        attribute_2 = extract_asymmetry(image, ax)
        push!(dataset, [attribute_1, attribute_2])
    end
    return hcat(dataset...)'
end

function extraer_caracteristicas_aprox2(inputs_bmp::Vector{Matrix{Float64}}, inputs_gray::Vector{Matrix{Float64}}, axis::Vector{Any}) 
    dataset = []
    for (image_binary, image_gray, ax) in zip(inputs_bmp, inputs_gray, axis)
    attribute_1 = extract_border_length(image_binary)
    attribute_2 = extract_asymmetry(image_binary, ax)
    attribute_3 = extract_mean_pixels_value(image_gray, image_binary)

    push!(dataset, [attribute_1, attribute_2, attribute_3])
    end
    return hcat(dataset...)'
end

function extraer_caracteristicas_aprox3(inputs_bmp::Vector{Matrix{Float64}}, inputs_gray::Vector{Matrix{Float64}}, inputs_color::Vector{Any}, axis::Vector{Any}) 
    dataset = []
    for (image_binary, image_gray, image_color, ax) in zip(inputs_bmp, inputs_gray, inputs_color, axis)
    attribute_1 = extract_border_length(image_binary)
    attribute_2 = extract_asymmetry(image_binary, ax)
    attribute_3 = extract_mean_pixels_value(image_gray, image_binary)
    attribute_4 = extract_mean_color_pixels_value(image_color, image_binary)
    
    push!(dataset, [attribute_1, attribute_2, attribute_3, attribute_4])
    end
    return hcat(dataset...)'
end