import numpy as np
import pandas as pd
import random


acceptablevalue=0.5
eta=0.5

def readfile():# read file line by line
    file=open("readfile.txt")
    lineone=file.readline()
    list_read_line_one=lineone.split() # we make first linr spilting to make it easier with indicies
    m=int(list_read_line_one[0])
    l=int(list_read_line_one[1])
    n=int(list_read_line_one[2])
    k=int(file.readline())
    matrx=np.zeros((k,(m+n))) # iniliztion matrix to store x and y in matrix
    for i in range(k):
        temp_list=file.readline()
        converted_list=temp_list.split()
        for j in range (len(converted_list)):
            converted_list[j]=float(converted_list[j]) # we convert it to float to make it eaiser to store x from it
        matrx[i]=converted_list
    return  matrx,m,n,l,k
read_matrix,m,n,l,k=readfile()
#print(read_matrix," ",m," ",n," ",l)
class node: # we make class node for each neuron in each layer to store the information for every node
    def __init__(self):
        self.name =" "
        self.input=0
        self.output=0
        self.error = 0
        self.list_weight=[]
        self.list_weight_updated=[]
        self.target=0

def sigmoid(f):#sigmoid function
    return 1 / (1 + np.exp(-f))

def weightslist(m,l,n):# inilize the weigthts random to our network
    listweights=[]
    for i in range((m*l)+(l*n)):
        r=random.uniform(-2,2)
        listweights.append(r)
    return listweights


def MSE(list_output_layer,n):#claculate the meean square error
    EP=0
    for i in range(n) :
        EP+=np.power(list_output_layer[i].target-list_output_layer[i].output,2)
    EP=EP*0.5

    return EP

def feedforward_inside_back_propagation(list_input_layer,list_hidden_layer,list_output_layer):
    m=len(list_input_layer)
    l=len(list_hidden_layer)
    n=len(list_output_layer)
    for i in range(m):
        for j in range(l):
            list_hidden_layer[j].input+=list_input_layer[i].output *list_input_layer[i].list_weight[j]
        #print("list hidden layer[1]:",list_hidden_layer[1].input)
    for s in range(l):
        # print("1",list_hidden_layer[s].input)
        # print("2",list_hidden_layer[s].output)

          list_hidden_layer[s].input = np.clip(list_hidden_layer[s].input, -709.78, 709.78)
          list_hidden_layer[s].output = sigmoid(list_hidden_layer[s].input)
    for i in range(l):
        for j in range(n):
            list_output_layer[j].input+=list_hidden_layer[i].output *list_hidden_layer[i].list_weight[j]
    for s in range(l):
            list_output_layer[s].input = np.clip(list_output_layer[s].input, -709.78, 709.78)
            list_output_layer[s].output = sigmoid(list_output_layer[s].input)
    #for i in range(n):
      #print(list_output_layer[i].output)
      #return list_output_layer[i].output
    return list_input_layer,list_hidden_layer,list_output_layer

#weights=[0.5,0.3,-0.8,0.9,1.1,0.8,-0.9,-1.0]
weights=weightslist(m,l,n)
#print(weights)
listfeedforward = []
# for i in range(k):
#    listfeedforward.append(feedforward_inside_back_propagation(read_matrix[i],m,n,l,weights))

def feed_forward_from_file(example,m,l,n,list_weight_from_file):
    list_y = []
    print("m",m,"l",l,"n",n)
    for i in range (n,m+n):  #for target y
        list_y.append(example[i])
        #print(list_y)
    list_input_layer = []
    for i in range(m):  # for input layer
        x = node()
        x.output = example[i]
        list_input_layer.append(x)
    list_hidden_layer = []
    for i in range(l):  # for hidden layer
        x = node()
        list_hidden_layer.append(x)
    list_output_layer = []
    for i in range(n):  # for output layer
        x = node()
        x.target=list_y[i]
        list_output_layer.append(x)
    k = 0
    for i in range(m):
        for j in range(l):
            list_input_layer[i].list_weight.append(list_weight_from_file[k])
            k += 1
    for i in range(l):
        for j in range(n):
            list_hidden_layer[i].list_weight.append(list_weight_from_file[k])
            k += 1
    for p in range(m):
        for q in range(l):
            list_hidden_layer[q].input += list_input_layer[p].output * (list_input_layer[p].list_weight[q])
        print("weigths:", list_input_layer[0].list_weight[0])
        print("list:", list_input_layer[0].output)
        print("list2:", list_input_layer[0].input)
        # print("list hidden layer[1]:",list_hidden_layer[1].input)
    for s in range(l):
        list_hidden_layer[s].output = sigmoid(list_hidden_layer[s].input)
    for i in range(l):
        for j in range(n):
            list_output_layer[j].input += list_hidden_layer[i].output * list_hidden_layer[i].list_weight[j]
    for s in range(l):
        list_output_layer[s].output = sigmoid(list_output_layer[s].input)
    for i in range(n):
        print(list_output_layer[i].output)
    # return list_output_layer[i].output

def Back_propagation(example,m,n,l):
    read_file = open("output.txt", "w") # to write the final wigths in the file

    weights = weightslist(m, l, n)
    list_y=[]

    for i in range (m,m+n):  #for target y
        list_y.append(example[i])
    list_input_layer = []
    for i in range(m):  # for input layer
        x = node()
        x.output = example[i]
        list_input_layer.append(x)
    list_hidden_layer = []
    for i in range(l):  # for hidden layer
        x = node()
        list_hidden_layer.append(x)
    for j in range(m):
        for k in range(l):
            list_input_layer[j].list_weight_updated.append(0)
    for h in range(l):
        for c in range(n):
            list_hidden_layer[h].list_weight_updated.append(0)
    list_output_layer = []
    for i in range(n):  # for output layer
        x = node()
        x.target=list_y[i]
        list_output_layer.append(x)
    k = 0
    for i in range(m):
        for j in range(l):
            list_input_layer[i].list_weight.append(weights[k])
            k += 1
    for i in range(l):
        for j in range(n):
            list_hidden_layer[i].list_weight.append(weights[k])
            k += 1

    (list_input_layer,list_hidden_layer,list_output_layer)=feedforward_inside_back_propagation(list_input_layer,list_hidden_layer,list_output_layer)

    #print("out:",list_hidden_layer[0].list_weight)
    ep=MSE(list_output_layer,n)
    print("ep:",ep)
    if ep>acceptablevalue:
         for i in range(50000):
             for j in range(n):  #Error of output neuron
                 list_output_layer[j].error=list_output_layer[j].output* (1-list_output_layer[j].output)*(list_output_layer[j].target-list_output_layer[j].output)
             #iterator=(m*l)
             for k in range(l):  #change output layer weight
                 for r in range(n):
                     list_hidden_layer[k].list_weight_updated[r]=list_hidden_layer[k].list_weight[r]+(eta*list_output_layer[r].error*list_hidden_layer[k].output)
             for i in range(l):   #calcolate back propagation hidden layer errors
                 temp=0
                 for j in range(n):
                     temp+=list_hidden_layer[i].list_weight[j] * list_output_layer[j].error
                 list_hidden_layer[i].error=list_hidden_layer[i].output *(1-list_hidden_layer[i].output)

             for w in range(m):   #change hidden layer weights (here)
                 for p in range(l):
                     list_input_layer[w].list_weight_updated[p]=list_input_layer[w].list_weight[p]+(eta*list_hidden_layer[p].error*list_input_layer[w].output)

             # to append the new weights in the list_wights
             for o in range(m):
                 for v in range(l):
                     list_input_layer[o].list_weight[v]=list_input_layer[o].list_weight_updated[v]
             for i in range(l):
                 for j in range(n):
                     list_hidden_layer[i].list_weight[j]=list_hidden_layer[i].list_weight_updated[j]

             (list_input_layer,list_hidden_layer,list_output_layer)=feedforward_inside_back_propagation(list_input_layer,list_hidden_layer,list_output_layer)
         for i in range(m):
             for k in range (l):


                read_file.write(str(list_input_layer[i].list_weight[k]))
                read_file.write('  ')
         for j in range(l):

             for r in range(n):

                 read_file.write(str((format(list_hidden_layer[j].list_weight[r],".2f"))))
    print("MSE:",MSE(list_output_layer,n))
    read_file.close()
#print("readmtrx0:",read_matrix[0])
#print("m:",m," ",n," ",l)
file_weight_update=open("output.txt","r")
list_weight_from_file=file_weight_update.readline()
list_weight_from_file_spite=list_weight_from_file.split()
list_weight_from_file_int=[]
print(list_weight_from_file_spite)
for i in range (len(list_weight_from_file)):
     temp=float(list_weight_from_file_spite[i])
     list_weight_from_file_int.append(temp)
for k in range(m):
     Back_propagation(read_matrix[k],m,n,l)
     feed_forward_from_file(read_matrix[k],m,l,n,list_weight_from_file_int)