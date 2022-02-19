import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np


def OCT_rendering(layers, atten_s, intens, Bz=5):
    H = 64
    W = 64

    I0 = 400
    atten_s = (0 - atten_s) / 400 * 1024 / H
    layers = layers * H
    intens = intens * I0  # un normalized
    intens = intens.view(Bz, 1, W)
    layer_num = 4
    uii = -1 / 900 * 1024 / H
    # remember to switch to cuda tensors  !!!
    # A  = torch.zeros([bz, 4,300], dtype=torch.float)
    # layers = torch.zeros([bz, 4,300], dtype=torch.float)
    # onelayer    = torch.zeros([1, 4,1], dtype=torch.float)
    # onelayer [0,:,0]=torch.tensor( [50,80,150,200] )

    # layers [0:5,:,0:300]    =   onelayer
    # print(layers)

    # attenuats = torch.ones([5, 4,300], dtype=torch.float)
    # attenuats  = attenuats*(-1/500)

    # atten_s =  torch.zeros([5, 4,300] , dtype=torch.float)
    out = torch.zeros([Bz, H, W], dtype=torch.float)
    line = torch.zeros([1, H, 1], dtype=torch.float)
    x2 = torch.arange(0, H)
    line[0, :, 0] = x2
    out[0:Bz, :, 0:W] = line  # this a template
    # out  =  torch.exp (out*(-1/500))

    signal_extend = torch.ones([layer_num, Bz, H, W], dtype=torch.float)

    for i in range(layer_num):  # generate the plane signal without diffetent layer atenuation
        # bias with layer psotion for attenuation start

        sub_tractor1 = torch.zeros([Bz, H, W], dtype=torch.float)
        boundi = torch.ones([Bz, 1, W], dtype=torch.float)
        boundi[:, 0, :] = layers[:, i, :]  # always use this to keep the sha[pe
        sub_tractor1[:, 0:H, :] = boundi  # copy the bound dary  to 0:300 in H
        # Ii = I0 * np.exp((bound_pos[i] - bound_pos[0] ) * uii)
        Ii = torch.ones([Bz, 1, W], dtype=torch.float)
        Ii[:, 0, :] = layers[:, i, :] - layers[:, 0, :]  # use the whole depth to de temin the attenua of peaks
        Ii = Ii.cuda()
        Ii = intens * torch.exp(Ii * uii)
        Ii_exten = torch.ones([Bz, H, W], dtype=torch.float)  # extend Ii at each H  position
        Ii_exten[:, 0:H, :] = Ii
        # print(Ii_exten)

        x = out - sub_tractor1

        attenuation_i = torch.ones([Bz, 1, W], dtype=torch.float)
        attenuation_i[:, 0, :] = atten_s[:, i, :]  # use the whole depth to de temin the attenua of peaks
        # extensde it to the whole line
        attenuation_i_ex = torch.ones([Bz, H, W], dtype=torch.float)
        attenuation_i_ex[:, 0:H, :] = attenuation_i

        signals = Ii_exten * torch.exp(x * attenuation_i_ex)
        # print(signals)
        signal_extend[i, :, :, :] = signals
        # print(signal_extend)

        pass
    # genetate the windwo to piece wise select
    # the last window is always the same
    window_extend = torch.ones([layer_num, Bz, H, W], dtype=torch.float)
    windows = torch.ones([Bz, H, W], dtype=torch.float)
    layers = layers.type(torch.IntTensor)
    # every layer need to mask the front part :
    for i in range(layer_num):
        for j in range(Bz):
            for k in range(W):
                window_extend[i, j, 0:layers[j, i, k], k] = 0
    # print(window_extend)
    # except the  final layer, the  belw part of the layer should be masked:
    for i in range(layer_num - 1):
        for j in range(Bz):
            for k in range(W):
                window_extend[i, j, layers[j, i + 1, k]: H, k] = 0
    # print(window_extend)

    sum = torch.zeros([Bz, H, W], dtype=torch.float)
    for i in range(layer_num):
        sum = sum + signal_extend[i, :, :, :] * window_extend[i, :, :, :]
    sum = sum + 30
    # one = sum[0,:,:]
    # dis = one.cpu().detach().numpy()
    # cv2.imshow('Deeplearning one',dis.astype(np.uint8))
    sum = sum.cuda()
    return sum


def layers_visualized(layers, H):
    bz, layer_n, W = layers.size()
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.IntTensor)
    layers = torch.clamp(layers, 0, H - 1)

    # out depth = 1
    out = torch.zeros([bz, 1, H, W], dtype=torch.float)
    # every layer need to mask the front part :
    # for i in range(layer_n):
    for j in range(bz):
        for k in range(W):
            out[j, 0, layers[j, 0, k], k] = 0.5
            out[j, 0, layers[j, 1, k], k] = 1

    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def boundary_visualized_integer_encodeing(layers, H):  # this is for sheath contour segmentation
    # inetgral encoding return a  one channel map with different numers
    bz, layer_n, W = layers.size()
    intial_layers = layers
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.LongTensor)
    layers = torch.clamp(layers, 0, H - 1)
    # out depth = 1
    # out  = torch.zeros([bz,1, H,W], dtype=torch.float) #  the back ground is zero
    out = torch.zeros([bz, 1, H, W], dtype=torch.float)  # Assign boundary value to one channel

    # every layer need to mask the front part :
    # for i in range(layer_n):
    for j in range(bz):
        for i in range(layer_n):
            out[j, 0, layers[j, i], torch.arange(W)] = (i + 1) / layer_n  # assign the same value a same continue layer
            # out[j, 0, layers[j, i],torch.arange(W)] = 1 # assign the same value a same continue layer
    out = out.type(torch.LongTensor)

    out = out.cuda()

    return out


def boundary_visualized_onehot_encodeing(layers, H):  # this is for sheath contour segmentation
    # inetgral encoding return a  one channel map with different numers
    bz, layer_n, W = layers.size()
    intial_layers = layers
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.LongTensor)
    layers = torch.clamp(layers, 0, H - 1)
    # out depth = 1
    # out  = torch.zeros([bz,1, H,W], dtype=torch.float) #  the back ground is zero
    out = torch.zeros([bz, layer_n, H, W], dtype=torch.float)  # Assign boundary value to one channel

    # every layer need to mask the front part :
    # for i in range(layer_n):
    for j in range(bz):
        for i in range(layer_n):
            # out[j, 0, layers[j, i],torch.arange(W)] = (i+1)/layer_n # assign the same value a same continue layer
            # out[j, 0, layers[j, i],torch.arange(W)] = 1 # assign the same value a same continue layer
            out[j, i, layers[j, i], torch.arange(W)] = 1  # assign value to a individul channnel
            out[j, i, layers[j, i] - 1, torch.arange(W)] = 1  # assign value to a individul channnel
            out[j, i, layers[j, i] - 2, torch.arange(W)] = 1  # assign value to a individul channnel

    out = out.type(torch.FloatTensor)

    out = out.cuda()

    return out


# This the order of the full list for the IVUS annotation
# 'catheter': ['1', 'catheter', 'test'],
#            'lumen': ['2', 'vessel', 'lumen'],
#            'wire': ['3','guide-wire', 'guidewire'],
#            'media': ['4','vessel (media)', 'vessel(media)', 'media'],
#            'branch': ['5','vessel(side-branch)', 'vessel (side-branch)', 'vessel(sidebranch)', 'vessel (sidebranch)',
#                       'side-branch', 'sidebranch', 'bifurcation'],
#            'stent': ['6','stent'],
#            'plaque': ['7','plaque'],
#            'calcium': ['8','calcification', 'calcium'],


# encode as sigle channel inte with up and lowe boudnary
def layers_visualized_integer_encodeing_full(layers,existence, H,reverse_exs=True):  # this is for sheath contour segmentation
    # inetgral encoding return a  one channel map with different numers
    if (reverse_exs == True):
        existence = 1- existence
    bz, layer_n, W = layers.size()
    intial_layers = layers
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.LongTensor)
    layers = torch.clamp(layers, 0, H - 1)
    # out depth = 1
    out = torch.zeros([bz, 1, H, W], dtype=torch.float)  # the back ground is zero,
    # every layer need to mask the front part :
    for l in range(0, layer_n, 2):  # the range step of the layer is 2 since every object have up and lower
        # confine up and lowerbound
        layers[:, l + 1, :] = torch.where(layers[:, l + 1, :] < layers[:, l, :], layers[:, l, :], layers[:, l + 1, :])
        for j in range(bz):
            this_layer_x = torch.arange(0, W)
            combine_exist = existence[j,l, :] * existence[j,l+1, :]
            presence_x  = this_layer_x[torch.where(combine_exist >0.5)[0]]


            for i in presence_x:
                out[j, 0, torch.arange(layers[j,l,i],layers[j,l+1,i]), i] =  (l + 2) / layer_n
    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def layers_visualized_integer_encodeing(layers, H):  # this is for sheath contour segmentation
    # inetgral encoding return a  one channel map with different numers
    bz, layer_n, W = layers.size()
    intial_layers = layers
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.IntTensor)
    layers = torch.clamp(layers, 0, H - 1)
    # out depth = 1
    out = torch.zeros([bz, 1, H, W], dtype=torch.float)  # the back ground is zero
    # every layer need to mask the front part :
    # for i in range(layer_n):
    for j in range(bz):
        for k in range(W):
            mean0 = torch.mean(intial_layers[j, 0, :])
            mean1 = torch.mean(intial_layers[j, 1, :])

            if (mean0 < mean1):
                out[j, 0, 0:layers[j, 0, k], k] = 0.5  # first layer is the sheath, albels 0:layer is 0.5
                out[j, 0, layers[j, 1, k]:H, k] = 1  # second layer is the contou, albels is 1
            else:
                out[j, 0, 0:layers[j, 1, k], k] = 0.5  # first layer is the sheath, albels 0:layer is 0.5
                out[j, 0, layers[j, 0, k]:H, k] = 1  # second layer is the contou, albels is 1
    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def layers_visualized_OneHot_encodeing(layers, H):  # this is for sheath contour segmentation
    # inetgral encoding return a  one channel map with different numers
    bz, layer_n, W = layers.size()
    # layers = layers +0.1
    layers = layers * H
    layers = layers.type(torch.IntTensor)
    layers = torch.clamp(layers, 0, H - 1)
    # out depth = 3
    out = torch.zeros([bz, 3, H, W], dtype=torch.float)  # has three channels of output, the none target area is zero
    # every layer need to mask the front part :
    # for i in range(layer_n):
    for j in range(bz):
        for k in range(W):
            out[j, 0, layers[j, 0, k]:layers[j, 1, k], k] = 1  # first chnnel is the backgrond
            # out[j,0,layers[j,1,k]:H,k]=1 #

            out[j, 1, 0:layers[j, 0, k], k] = 1  # second channel is the sheath, albels is 1

            out[j, 2, layers[j, 1, k]:H, k] = 1  # third  channel is the tisue, albels is 1

    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def integer2onehot(integer):
    bz, c_n, H, W = integer.size()
    # layers = layers +0.1
    # layers   =  layers * H
    # layers = layers.type(torch.IntTensor)
    # layers = torch.clamp(layers, 0, H-1)
    # out depth = 3
    out = torch.zeros([bz, 3, H, W], dtype=torch.float)  # has three channels of output, the none target area is zero
    # every layer need to mask the front part :
    # for i in range(layer_n):

    out[:, 0, :, :] = integer[:, 0, :, :] < 0.25  # first chnnel is the backgrond
    # out[j,0,layers[j,1,k]:H,k]=1 #

    out[:, 1, :, :] = (integer[:, 0, :, :] > 0.25) * (
                integer[:, 0, :, :] < 0.75)  # second channel is the sheath, albels is 1

    out[:, 2, :, :] = integer[:, 0, :, :] > 0.75  # third  channel is the tisue, albels is 1
    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def onehot2integer(onehot):
    bz, c_n, H, W = onehot.size()
    # layers = layers +0.1
    # layers   =  layers * H
    onehot = onehot.type(torch.FloatTensor)
    onehot = onehot.cuda()

    # layers = torch.clamp(layers, 0, H-1)
    # out depth = 3
    out = torch.zeros([bz, 1, H, W], dtype=torch.float)  # has three channels of output, the none target area is zero
    out = out.cuda()

    # every layer need to mask the front part :
    # for i in range(layer_n):
    for i in range(c_n):
        out[:, 0, :, :] = out[:, 0, :, :] + onehot[:, i, :, :] * (i + 1) / c_n

    # out   =( out  -0.5)/0.5
    out = out.cuda()
    return out


def onehot2layers(onehot):
    C, H, W = onehot.size()
    layer1 = np.zeros(W)
    layer2 = np.ones(W) * (H - 1)
    for i in range(W):  # search the sheath contour  # second channel is the sheath
        for j in range(H - 5):
            if (onehot[1, j, i] > 0.5 and onehot[1, j + 2, i] < 0.5):
                layer1[i] = j
                break
    for i in range(W):  # search the tissue contour
        for j in range(H - 5):
            if (onehot[2, j, i] < 0.5 and onehot[2, j + 2, i] > 0.5):
                layer2[i] = j
                break
    return layer1, layer2

def onehot2layers_cut_bound(onehot): # cut the up and lower boundarys
    C, H, W = onehot.size()
    layers= torch.ones([2*(C-1), W], dtype=torch.long)*(H - 1)
    layers =layers.cuda()
    # layer2 = np.ones(W) * (H - 1)
    for k in range(0,C-1):
        for i in range(W):  # search the sheath contour  # second channel is the sheath
            for j in range(H - 5): # ignore the first background layer
                if (onehot[k+1, j, i] < 0.5 and onehot[k+1, j + 2, i] > 0.5):
                    layers[2*k,i] = j
                    break
            for j in range(layers[2*k,i],H - 5):  # ignore the first background layer
                if (onehot[k + 1, j, i] > 0.5 and onehot[k + 1, j + 2, i] < 0.5):
                    layers[2 * k+1, i] = j
                    break

    # out = out.type(torch.FloatTensor)
    layers=layers.type(torch.FloatTensor)/(H-1)
    layers =layers.cuda()

    return layers
def pytorch_test():
    H = 300
    I0 = 400
    layer_num = 4
    uii = -1 / 900
    # remember to switch to cuda tensors  !!!
    Intens = torch.zeros([5, 1, 300], dtype=torch.float)
    Intens[0:5, 0, :300] = I0
    A = torch.zeros([5, 4, 300], dtype=torch.float)
    layers = torch.zeros([5, 4, 300], dtype=torch.float)
    onelayer = torch.zeros([1, 4, 1], dtype=torch.float)
    onelayer[0, :, 0] = torch.tensor([50, 80, 150, 200])

    layers[0:5, :, 0:300] = onelayer
    print(layers)

    attenuats = torch.ones([5, 4, 300], dtype=torch.float)
    attenuats = attenuats * (-1 / 500)

    atten_s = torch.zeros([5, 4, 300], dtype=torch.float)
    out = torch.zeros([5, 300, 300], dtype=torch.float)
    line = torch.zeros([1, 300, 1], dtype=torch.float)
    x2 = torch.arange(0, H)
    line[0, :, 0] = x2
    out[0:5, :, 0:300] = line  # this a template
    # out  =  torch.exp (out*(-1/500))

    signal_extend = torch.ones([layer_num, 5, 300, 300], dtype=torch.float)

    for i in range(layer_num):  # generate the plane signal without diffetent layer atenuation
        # bias with layer psotion for attenuation start
        Bz = 5
        sub_tractor1 = torch.zeros([5, 300, 300], dtype=torch.float)
        boundi = torch.ones([5, 1, 300], dtype=torch.float)
        boundi[:, 0, :] = layers[:, i, :]  # always use this to keep the sha[pe
        sub_tractor1[:, 0:300, :] = boundi  # copy the bound dary  to 0:300 in H
        # Ii = I0 * np.exp((bound_pos[i] - bound_pos[0] ) * uii)
        Ii = torch.ones([5, 1, 300], dtype=torch.float)
        Ii[:, 0, :] = layers[:, i, :] - layers[:, 0, :]  # use the whole depth to de temin the attenua of peaks
        Ii = Intens * torch.exp(Ii * uii)
        Ii_exten = torch.ones([5, 300, 300], dtype=torch.float)  # extend Ii at each H  position
        Ii_exten[:, 0:300, :] = Ii
        print(Ii_exten)

        x = out - sub_tractor1

        attenuation_i = torch.ones([5, 1, 300], dtype=torch.float)
        attenuation_i[:, 0, :] = attenuats[:, i, :]  # use the whole depth to de temin the attenua of peaks
        # extensde it to the whole line
        attenuation_i_ex = torch.ones([5, 300, 300], dtype=torch.float)
        attenuation_i_ex[:, 0:300, :] = attenuation_i

        signals = Ii_exten * torch.exp(x * attenuation_i_ex)
        print(signals)
        signal_extend[i, :, :, :] = signals
        print(signal_extend)

        pass
    # genetate the windwo to piece wise select
    # the last window is always the same
    window_extend = torch.ones([layer_num, 5, 300, 300], dtype=torch.float)
    windows = torch.ones([5, 300, 300], dtype=torch.float)
    layers = layers.type(torch.IntTensor)
    # every layer need to mask the front part :
    for i in range(layer_num):
        for j in range(Bz):
            for k in range(300):
                window_extend[i, j, 0:layers[j, i, k], k] = 0
    print(window_extend)
    # except the  final layer, the  belw part of the layer should be masked:
    for i in range(layer_num - 1):
        for j in range(Bz):
            for k in range(300):
                window_extend[i, j, layers[j, i + 1, k]: 300, k] = 0
    print(window_extend)

    sum = torch.zeros([5, 300, 300], dtype=torch.float)
    for i in range(layer_num):
        sum = sum + signal_extend[i, :, :, :] * window_extend[i, :, :, :]

    one = sum[0, :, :] + 80
    dis = one.cpu().detach().numpy()
    cv2.imshow('Deeplearning one', dis.astype(np.uint8))
    return out


if __name__ == '__main__':
    pytorch_test()
    pytorch_test()



