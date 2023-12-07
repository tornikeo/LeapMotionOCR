clear all; close all; clc;

testfiles = dir('training_data/*.csv');
for k = 1:length(testfiles)
    testdata  = readmatrix(strcat("training_data/", testfiles(k).name));
    C = digit_classify(testdata);
    disp(["Predicted ", int2str(C), " for file", testfiles(k).name]);
end