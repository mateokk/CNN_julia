using MLDatasets
using LinearAlgebra
using Random

x_train, y_train = MNIST(split = :train)[:];
x_test,  y_test  = MNIST(split = :test)[:];

mutable struct weights
    conv
    first
    output    
end
mutable struct bias
    conv
    first
    output    
end

################
# PARAMETERS
###############
f = 8  # Number of filters
pool_param = [2, 2, 2] # Height, width, stride
global α = 0.003
global epochs = 5
w = weights(randn(3, 3, f), randn(300, 14*14*f), randn(10, 300))
b = bias(randn(28, 28, f), randn(300, 1), randn(10, 1))

function addPad(x, P)
    H, W = size(x)
    B = zeros(H + 2P, W + 2P)
    for i=1:W, j=1:H
        B[j+P, i+P] = x[j, i]
    end
    return B
end

function conv_forward(x_in, w, b) 
    H, W = size(x_in)           
    HH, WW, D = size(w)         
    P = 1                       # Padding
    x = addPad(x_in, P)

    H_R = 1 + H - HH + 2P
    W_R = 1 + W - WW + 2P
    out = zeros(H_R, W_R, D)
    c = size(x, 1) - HH + 1
    d = size(x, 2) - WW + 1
    for i=1:c, j=1:d, k=1:D
        out[i, j, k] = sum(x[i:i+HH-1, j:j+WW-1] .* w[:, :, k]) .+ b[i, j, k]
    end
    cache = (x_in, w, b)
    return out, cache
end
function conv_backward(dout, cache)
    x, w, b = cache
    HH, WW, D = size(w)
    P = 1                   
    x = addPad(x, P)
    dx = zeros(size(x))
    dw = zeros(size(w))

    c = size(x, 1) - HH + 1
    d = size(x, 2) - WW + 1
    for i=1:c, j=1:d, k=1:D
        dx[i:i+HH-1, j:j+WW-1] += dout[i, j, k] .* w[:, :, k]
    end
    dx = view(dx, P+1:size(dx, 1)-P, P+1:size(dx, 2)-P)  # removing padded rows and column
    for i=1:c, j=1:d, k=1:D
        dw[:, :, k] += dout[i, j, k] .* x[i:i+HH-1, j:j+WW-1]
    end
    db = dout
    return dx, dw, db
end

function max_pool_forward(x, pool_param)
    H, W, D = size(x)
    H_P = pool_param[1]
    W_P = pool_param[2]
    S = pool_param[3]
    HH = (Int)(1 + (H - H_P)/S)
    WW = (Int)(1 + (W - W_P)/S)
    out = zeros(HH, WW, D)
    c = H - H_P + 1
    d = W - W_P + 1
    for i=1:S:c, j=1:S:d, k=1:D
        out[floor(Int, i/S+1), floor(Int, j/S+1), k] = findmax(x[i:i+H_P-1, j:j+W_P-1, k])[1]
    end
    cache = x, pool_param
    return out, cache
end

function max_pool_backward(dout, cache)
    x, pool_param = cache
    S = pool_param[3]
    HH, WW, D = size(dout)
    dx = zeros(size(x))
    for i=1:HH, j=1:WW, k=1:D
        x_pool = x[S*i-S+1:S*i, S*j-S+1:S*j, k]
        mask = (x_pool .== findmax(x_pool)[1])
        dx[S*i-S+1:S*i, S*j-S+1:S*j, k] = mask .* dout[i, j, k]
    end
    return dx
end

function fc_forward(x, w, b)
    out = w * x .+ b
    cache = (x, w, b)
    return out, cache
end

function fc_backward(dout, cache)
    x, w, b = cache
    dx = w' * dout
    dw = dout * x'
    db = dout
    return dx, dw, db
end
function one_hot(y)
    one_hot_y = zeros(size(y, 1), 10)
    for i in size(y, 1)
        one_hot_y[i, y[i] + 1] = 1
    end
    one_hot_y = one_hot_y'
    return one_hot_y
end

function relu_forward(x)
    out = max.(x, 0)
    cache = x;
    return out, cache
end

function relu_backward(dout, cache)
    x = cache;
    dx = dout .* (x .>= 0)
    return dx
end

function softmax_loss(x, y)
    probs = exp.(x .- maximum(x))
    probs /= sum(probs)
    dout_x = probs
    dout_x[y] -= 1
    return dout_x
end
function flatten_forward(x, n)
    out = reshape(x, :, n)
    cache = x
    return out, cache
end
function flatten_backward(dout, cache)
    out = reshape(dout, size(cache))
    return out
end

function get_accuracy(predictions, y)
    len = size(y)[1]
    acc = sum(argmax(predictions[:, i])-1 == y[i] ? 1 : 0 for i in eachindex(y))/len * 100.
    return string("Accuracy: ", round(acc, digits=2), " %")
end
function sigmoid(x)
    out = 1.0 ./ (1.0 .+ exp.(-x))
    cache = x
    return out, cache
end
function sigmoid_backward(dout, cache)
   x = cache
   dx = dout .* (sigmoid(x)[1] .* (1 .- sigmoid(x)[1]))
   return dx
end


function predict(w, b, x_data, y_data, data_size, train="true")
    for e=1:epochs
        predictions = zeros(10, data_size)
        loss = 0
        for i=1:data_size
            x = x_data[:, :, i]
            y = y_data[i]
            y = one_hot(y)
            # FORWARD PASS
            out1, conv_cache = conv_forward(x, w.conv, b.conv)
            out2, mp_cache = max_pool_forward(out1, pool_param)
            out1, sigm1_cache = sigmoid(out2)

            out2, flat_cache = flatten_forward(out1, 1)
            out1, first_cache = fc_forward(out2, w.first, b.first)
            out2, relu1_cache = relu_forward(out1)
            out1, output_cache = fc_forward(out2, w.output, b.output)
            predictions[:, i] = out1
            dout = softmax_loss(out1, argmax(y)[1])
            if cmp(train, "true") == 0
                errors(dout, conv_cache, mp_cache, sigm1_cache, flat_cache, first_cache, relu1_cache, output_cache, i)
            end        
        end
        if cmp(train, "true") != 0
            a = get_accuracy(predictions, y_data[1:data_size])
            print("Validation results: ", a, "\n")
            return;
        end
        a = get_accuracy(predictions, y_data[1:data_size])
        print("Epoch: ", e, "/", epochs, ", ", a, "\n")
    end
end
function errors(dout, conv_cache, mp_cache, sigm1_cache, flat_cache, first_cache, relu1_cache, output_cache, e)
    # BACKWARD PASS
    δ, output_dw, output_db = fc_backward(dout, output_cache)
    δ = relu_backward(δ, relu1_cache)
    δ, first_dw, first_db = fc_backward(δ, first_cache)
    δ = flatten_backward(δ, flat_cache)
    δ = sigmoid_backward(δ, sigm1_cache)
    δ = max_pool_backward(δ, mp_cache)
    δ, conv_dw, conv_db = conv_backward(δ, conv_cache)
    w.conv = w.conv .- α .* conv_dw
    w.first = w.first .- α .* first_dw
    w.output = w.output .- α .* output_dw
    b.conv = α .* conv_db
    b.first = α .* first_db
    b.output = α .* output_db 
end
print("TRAINING\n")
predict(w, b, x_train, y_train, 5000)
print("VALIDATION\n")
predict(w, b, x_test, y_test, 10000, "false")

