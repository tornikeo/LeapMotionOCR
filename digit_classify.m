clear all; close all; clc;

C = magic(3);

digit_classify_here(C)

function result = digit_classify_here(C)
    py_c = py.numpy.array(C);
    result = pyrunfile("hello.py", "result", arr=C);
end