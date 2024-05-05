

ann1 = Chain(

    Conv((3, 3), 3=>16, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(3600, 3),

    softmax
)

ann2 = Chain(

    Conv((3, 3), 3=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), relu),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(1568, 3),

    softmax
)

ann3 = Chain(

    Conv((3, 3), 3=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 32=>32, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(288, 3),

    softmax
)

ann4 = Chain(

    Conv((3, 3), 3=>8, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 8=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 16=>32, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 32=>32, pad=(1,1), relu),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(32, 3),

    softmax
)

ann5 = Chain(

    Conv((3, 3), 3=>8, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 8=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 16=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(144, 3),

    softmax
)

ann6 = Chain(

    Conv((3, 3), 3=>8, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 8=>16, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(784, 3),

    softmax
)

ann7 = Chain(

    Conv((3, 3), 3=>8, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(1800, 3),

    softmax
)

ann8 = Chain(

    Conv((3, 3), 3=>4, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 4=>8, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 8=>8, pad=(1,1), relu),

    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),

    Dense(72, 3),

    softmax
)

ann9 = Chain(

    Conv((3, 3), 3=>4, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((3, 3), 4=>8, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(392, 3),

    softmax
)

ann10 = Chain(

    Conv((3, 3), 3=>4, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(900, 3),

    softmax
)

ann11 = Chain(

    Conv((5, 5), 3=>16, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((5, 5), 16=>32, pad=(1,1), relu),

    MaxPool((2,2)),

    Conv((5, 5), 32=>32, pad=(1,1), relu),

    MaxPool((2,2)),
    
    x -> reshape(x, :, size(x, 4)),

    Dense(128, 3),

    softmax
)

arquitecturas = [ann1, ann2, ann3, ann4, ann5, ann6, ann7, ann8, ann9, ann10, ann11]
