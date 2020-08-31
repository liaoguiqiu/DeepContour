import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
def OCT_rendering(layers,atten_s):


    pass

def pytorch_test():
    H  = 300 
    I0= 400
    layer_num   = 4 
    uii = -1/900
    # remember to switch to cuda tensors  !!!
    A  = torch.zeros([5, 4,300], dtype=torch.float)
    layers = torch.zeros([5, 4,300], dtype=torch.float)
    onelayer    = torch.zeros([1, 4,1], dtype=torch.float)
    onelayer [0,:,0]=torch.tensor( [50,80,150,200] ) 

    layers [0:5,:,0:300]    =   onelayer 
    print(layers)


    attenuats = torch.ones([5, 4,300], dtype=torch.float)
    attenuats  = attenuats*(-1/500)

    atten_s =  torch.zeros([5, 4,300] , dtype=torch.float)
    out  = torch.zeros([5, 300,300], dtype=torch.float)
    line = torch.zeros([1, 300,1],dtype=torch.float)
    x2 =  torch.arange(0, H) 
    line[0,:,0]  = x2
    out[0:5,:,0:300]  = line  # this a template 
    #out  =  torch.exp (out*(-1/500))

    signal_extend = torch.ones([layer_num,5, 300,300], dtype=torch.float)


    for i in range(layer_num):  # generate the plane signal without diffetent layer atenuation 
       # bias with layer psotion for attenuation start
        Bz = 5
        sub_tractor1 = torch.zeros([5, 300,300], dtype=torch.float)
        boundi= torch.ones([5, 1,300], dtype=torch.float)
        boundi[:, 0,:]  = layers[:,i,:]  # always use this to keep the sha[pe
        sub_tractor1 [:,0:300,:]    =  boundi # copy the bound dary  to 0:300 in H 
        #Ii = I0 * np.exp((bound_pos[i] - bound_pos[0] ) * uii)
        Ii  = torch.ones([5, 1,300], dtype=torch.float)
        Ii[:, 0,:]  = layers[:,i,:]  - layers[:,0,:]  # use the whole depth to de temin the attenua of peaks 
        Ii = I0 * torch.exp (Ii* uii) 
        Ii_exten =  torch.ones([5, 300,300], dtype=torch.float)   # extend Ii at each H  position 
        Ii_exten[:,0:300,:]   = Ii
        print(Ii_exten)

        x = out -sub_tractor1

        attenuation_i  = torch.ones([5, 1,300], dtype=torch.float)
        attenuation_i[:, 0,:]  = attenuats[:,i,:]  # use the whole depth to de temin the attenua of peaks 
        # extensde it to the whole line 
        attenuation_i_ex   = torch.ones([5, 300,300], dtype=torch.float)  
        attenuation_i_ex[:,0:300,:]   = attenuation_i


        signals =  Ii_exten* torch.exp(x*attenuation_i_ex)
        print(signals)
        signal_extend[i,:,:,:]  = signals
        print(signal_extend)
        
        pass
    # genetate the windwo to piece wise select 
        # the last window is always the same 
    window_extend = torch.ones([layer_num,5, 300,300], dtype=torch.float)
    windows  = torch.ones([5, 300,300], dtype=torch.float)
    layers = layers.type(torch.IntTensor)
    # every layer need to mask the front part :
    for i in range(layer_num): 
        for j in range(Bz):
            for k in range(300):
                window_extend[i,j,0:layers[j,i,k],k]=0
    print(window_extend)
    # except the  final layer, the  belw part of the layer should be masked:
    for i in range(layer_num-1): 
        for j in range(Bz):
            for k in range(300):
                window_extend[i,j, layers[j,i+1,k]: 300,k]=0
    print(window_extend)

    sum = torch.zeros([5, 300,300], dtype=torch.float)
    for i in range(layer_num): 
        sum = sum+signal_extend[i,:,:,:]* window_extend[i,:,:,:]

    one = sum[0,:,:]
    dis = one.cpu().detach().numpy() 
    cv2.imshow('Deeplearning one',dis.astype(np.uint8)) 
    return out


if __name__ == '__main__':
     pytorch_test()
     pytorch_test()



         