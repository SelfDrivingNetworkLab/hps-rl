import torch


def mean_squared_loss(target, output):
    loss = torch.nn.MSELoss()
    return loss(output, target)


def mean_absolute_loss(output, target):
    # reduction can be none|sum|mean
    loss = torch.nn.L1Loss()
    return loss(output, target)


def huber_loss(output, target):
    # also known as smooth l1
    loss = torch.nn.SmoothL1Loss()
    return loss(output, target)


def cross_entropy_loss(output, target):
    # It is useful when training a classification problem with C classes.
    # input has to be a Tensor of size either (minibatch,C)(minibatch, C)(minibatch,C) or
    # (minibatch,C,d1,d2,...,dK)(minibatch, C, d_1, d_2, ..., d_K)(minibatch,C,d1​,d2​,...,dK​
    # a class index in the range [0,C−1] as the target for each value of a 1D tensor of size minibatch
    loss = torch.nn.CrossEntropyLoss()
    return loss(output, target)


def time_series_loss(output, target):
    # The Connectionist Temporal Classification loss.
    # Calculates loss between a continuous (unsegmented) time series and a target sequence.
    # CTCLoss sums over the probability of possible alignments of input to target,
    # producing a loss value which is differentiable with respect to each input node.
    # A. Graves et al.: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
    # https://www.cs.toronto.edu/~graves/icml_2006.pdf
    loss = torch.nn.CTCLoss()
    return loss(output, target)


def negative_log_likelihood_loss(output, target):
    # It is useful to train a classification problem with C classes.
    # The input given through a forward call is expected to contain log-probabilities of each class.
    # input has to be a Tensor of size either (minibatch,C)(minibatch, C)(minibatch,C)
    # Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer
    # in the last layer of your network. Cross entropy, itself, calculates softmax
    # The target that this loss expects should be a class index in the range [0,C−1] where C = number of classes;
    # Please do not forget to use torch.nn.LogSoftmax at the last layer
    loss = torch.nn.NLLLoss()
    return loss(output, target)


def kullback_leibler_divergence_loss(output, target):
    # KL divergence is a useful distance measure for continuous distributions and is often useful
    # when performing direct regression over the space of (discretely sampled) continuous output distributions.
    # the input given is expected to contain log-probabilities apply LogSoftmax,
    # The targets are interpreted as probabilities one-hot vector
    loss = torch.nn.KLDivLoss()
    return loss(output, target)

def bce_with_logit_loss(output,target):
    #This loss combines a Sigmoid layer and the BCELoss in one single class.
    # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as
    #targets are between 0 and 1
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(output,target)

def margin_ranking_loss(input,target):
    #input(N,D): N is batch size and D is size of sample
    #target is N
    #output is N
    loss = torch.nn.MarginRankingLoss()
    return loss(input,target)

def hinge_embedding_loss(output,target):
    #Measures the loss given an input tensor xxx and a labels tensor yyy (containing 1 or -1).
    #This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance as xxx ,
    # and is typically used for learning nonlinear embeddings or semi-supervised learning.
    #output any dimension
    #target any dimension
    loss = torch.nn.HingeEmbeddingLoss()
    return loss(output,target)

def multi_class_hinge_loss(output,target):
    #output(N,C): N batch size, C number of classes
    #target(N,C): same shape with output
    loss = torch.nn.MultiLabelMarginLoss()
    return loss(output,target)

def soft_margin_loss(output,target):
    #output: any dimension
    #target: any dimension (-1 or 1)
    loss = torch.nn.SoftMarginLoss()
    return loss(output,target)

def multi_class_soft_margin_loss(output,target):
    #multi-label one-versus-all loss based on max-entropy, between input xxx and target yyy of size (N,C)(N, C)(N,C)
    #both parameters (N,C)
    loss = torch.nn.MultiLabelSoftMarginLoss()
    return loss(output,target)

def cosine_embedding_loss(input1,input2,target):
    #This is used for measuring whether two inputs are similar or dissimilar, using the cosine distance,
    # and is typically used for learning nonlinear embeddings or semi-supervised learning.
    loss = torch.nn.CosineEmbeddingLoss()
    return loss(input1,input2,target)

def multi_margin_loss(output,target):
    #multi-class classification hinge loss (margin-based loss) between input xxx (a 2D mini-batch Tensor) and output yyy
    #(which is a 1D tensor of target class indices, 0≤y≤x.size(1)−10 \leq y \leq \text{x.size}(1)-10≤y≤x.size(1)−1 ):
    loss = torch.nn.MultiMarginLoss()
    return loss(output,target)
