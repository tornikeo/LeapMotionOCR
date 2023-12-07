One of the ideas how to approach the problem was to use some kind of feature extraction on the sets of points, which would describe the numbers in a way that makes the feature vectors easily distinguishable and categorizable. While the idea sounds good, a question arises: what kind of features? 

The numbers are described by consecutive points. The points itself have a location, which can be represented as a vector. Also every consecutive pair of points is important, because they form a line segment, which is part of the whole number. So what is important is location of points and form of a line segments. First thing we tried was to focus just on location of a points. But how to featurize that? 

We could for example divide the space into some discrete subspaces, what would 
