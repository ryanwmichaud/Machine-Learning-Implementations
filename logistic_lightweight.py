#Ryan Michaud
#10/03/2022
#This program reads in data from a csv and trains a logistic regression model to predict labels.
# This version uses less functions/objects 


import csv
import sys
import random
import math


def main(a1,a2,a3,a4,a5):
    #print(a1,a2,a3,a4,a5)
    data_set = a1
    learning_rate = float(a2)
    training_percent = float(a3)
    validation_percent = float(a4)
    seed = a5

    min_attribute_values = []
    max_attribute_values = []
    categories = []

    instances = []
    with open(data_set) as file:
        reader = csv.reader(file) 
        next(reader) #throw out header for now
        
        #read first line to start the max and min class variables
        line1 = next(reader)  
        num_attributes = 0
        
        for i in range(1,len(line1)):
            num_attributes += 1
            categories.append(None)
            try: 
                line1[i] = float(line1[i])  #make float if float
            except:
                possible_labels = set()     #track num categories if category
                possible_labels.add(line1[i])
                categories[i-1]=possible_labels #-1 bc categories doesnt have a label but inst does in index 0


            max_attribute_values.append(line1[i])   #get max and min started for each attribute
            min_attribute_values.append(line1[i])

        label = float(line1[0])
        attributes = line1[1:len(line1)]
        for i in range(len(attributes)):
            try: 
                attributes[i] = float(attributes[i])  #if float, convert it to one
            except:
                categories[i].add(attributes[i]) #if categorical, add to possible categories

        first_inst = [label]
        first_inst.extend(attributes)
        instances.append(first_inst)

        #add the rest of the instances finding the max and mins as we instantiate
        for line in reader:
            
            label = float(line[0])
            attributes = line[1:len(line)]

            for i in range(len(attributes)):
                try: 
                    attributes[i] = float(attributes[i])  #if float, convert it to one
                except:
                    categories[i].add(attributes[i]) #if categorical, add to possible categories

                #get max and mins for each attribute
                if attributes[i] < min_attribute_values[i]:
                    min_attribute_values[i] = attributes[i]
                if attributes[i] > max_attribute_values[i]:
                    max_attribute_values[i] = attributes[i]

            inst = [label]
            inst.extend(attributes)
            instances.append(inst)
        #closes file
    #scale them all now that we have the max and mins
    #print(Instance.max_attribute_values,Instance.min_attribute_values)
  
    for inst in instances:
        for i in range(1,num_attributes+1):
            try:
                #print('before', inst[i+1],'with max:',max_attribute_values[i],'and min:',min_attribute_values[i])
                inst[i+1] = (inst[i+1]-min_attribute_values[i])/(max_attribute_values[i]-min_attribute_values[i])
                #print('after', inst[i+1])
            except:
                pass
    #print (categories)

    #shuffle to keep data sets diverse
    random.seed(seed)
    random.shuffle(instances)
    #save our math to one time
    instances_length = len(instances)
    training_end = training_percent * instances_length
    validation_length = (validation_percent * instances_length)
    validation_end = validation_length + training_end

    #one hot
    new_instances = []
    for instance in instances:
        
        new_attributes = [] #gonna replace catagorical data w an attribute for each possible value for that attribute. one less really
        for i in range(num_attributes):
            
            if isinstance(instance[i+1],float):#if its a float just copy it. start at first attribute. 1th index after label          
                new_attributes.append(instance[i+1])
             
            else:       #if its categorical, one hot it 
                for label in categories[i]:    #will be the same across each time like a table
                    if label == instance[i+1]: #1 if its the one, 0 if its not
                        new_attributes.append(1)
                    else:                       
                        new_attributes.append(0)
                new_attributes.pop()             #drop the last column of each category. It'll be all 0s          
        #print (instance,'\nwith', categories)
        #instance.attributes=new_attributes
        instance  = instance[0:1]
        for a in new_attributes:
            instance.append(a)
        #print ('becomes',instance)
        new_instances.append(instance)
    instances = new_instances        
    


    training_set = []
    validation_set = []
    testing_set = []

    for i in range(instances_length): 
        if i <= training_end:
            training_set.append(instances[i])
        elif i <= validation_end:
            validation_set.append(instances[i])
        else:
            testing_set.append(instances[i])
        
    #print(len(instances),len(training_set),len(validation_set),len(testing_set))
    #print('mins',min_attribute_values, 'maxs', max_attribute_values)

    #initialze random weights btwn -0.1 and 0.1
    w0 = random.uniform(-0.1,0.1)
    num_attributes = len(instances[0]) -1   #-1 bc label isnt an attribute
    weights = []
    for i in range(num_attributes):
        weights.append(random.uniform(-0.1,0.1))
    
    accuracy = 0
    epochs = 0
    
    while accuracy <= 0.99 and epochs < 500:
        for instance in training_set:
           #linear regression prediction
            prediction = w0
            for i in range(num_attributes):
                prediction += weights[i]*instance[i+1] #+1 bc label
            #sigmoid
            prediction = 1/(1+math.e**(-1*prediction))
            
            #calculate changes in weights
            delta_w0 = -1*prediction*(1-prediction)*(instance[0] - prediction)
            delta_ws = []
            for i in range(num_attributes):
                delta_ws.append(-instance[i+1]*prediction*(1-prediction)*(instance[0] - prediction))
 
            #update weights
            w0 = w0 - learning_rate*delta_w0
            for i in range(num_attributes):
                #print(weights[i],end=' ')
                #print(weights[i],weights[i] -learning_rate*delta_ws[i])
                weights[i] = weights[i] - learning_rate*delta_ws[i]
                #print(weights[i])
            

        correct = 0
        for instance in validation_set:
            
           #linear regression prediction
            prediction = w0
            for i in range(num_attributes):
                prediction += weights[i]*instance[i+1]
            #sigmoid
            prediction = 1/(1+math.e**(-1*prediction))
            
            if round(prediction) == instance[0]:
                correct +=1
        accuracy = correct/validation_length

        print(epochs,accuracy)
        epochs += 1




    """
    #testing
    got_0 = 0
    got_1 = 0
    miss_0 =0
    miss_1 =0

    for instance in testing_set:
        #linear regression prediction
        prediction = w0
        for i in range(num_attributes):
            prediction += weights[i]*instance[i+1]
        #sigmoid
        prediction = 1/(1+math.e**(-1*prediction))
        
        if round (prediction) == 0:
            if instance[0] == 0:
                got_0 += 1
            else: 
                miss_0 += 1
        else:
            if instance[0] == 1:
                got_1 += 1
            else: 
                miss_1 += 1

                
        
        
               


    
    filename = 'results_'+str(data_set)+'_'+str(learning_rate)+'_'+str(seed)
    with open(filename, 'w') as output_file: 
        heading_writer = csv.writer(output_file, lineterminator=',\n')
        heading_writer.writerow([0,1])

        writer = csv.writer(output_file, lineterminator='\n')
        writer.writerow([got_0,miss_0,0])
        writer.writerow([miss_1,got_1,1])
        #closes file"""
    
            




if __name__ == "__main__" :
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
   