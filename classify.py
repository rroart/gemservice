import learntest as lt
import os
#import sys
import argparse

import torch
import torch.nn as nn
import numpy as np

import json
from datetime import datetime
from werkzeug.wrappers import Response
import shutil

from multiprocessing import Queue

import importlib

global dicteval
dicteval = {}
global dictclass
dictclass = {}
global dicttask
dicttask = {}
global count
count = 0

initial_tasks = 1

class Classify:
    def do_eval(self, request):
        global dicteval
        myobj = json.loads(request.get_data(as_text=True), object_hook=lt.LearnTest)
        print("geteval" + str(myobj.modelInt) + myobj.period + myobj.mapname)
        accuracy_score = dicteval[str(myobj.modelInt) + myobj.period + myobj.mapname]
        return Response(json.dumps({"accuracy": accuracy_score}), mimetype='application/json')

    def do_classify(self, request):
        dt = datetime.now()
        timestamp = dt.timestamp()
        myobj = json.loads(request.get_data(as_text=True), object_hook=lt.LearnTest)
        (config, model) = self.getmodel(myobj)
        (intlist, problist) = self.do_classifyinner(myobj, model)
        print(len(intlist))
        print(intlist)
        print(problist)
        dt = datetime.now()
        print ("millis ", (dt.timestamp() - timestamp)*1000)
        
        return Response(json.dumps({"classifycatarray": intlist, "classifyprobarray": problist }), mimetype='application/json')
        
    def do_classifyinner(self, myobj, model):
        array = np.array(myobj.classifyarray, dtype='f')
        intlist = []
        problist = []
        array = torch.FloatTensor(array)
        global dicttask
        task = dicttask[myobj.mapname]
        predictions = model(array, task)
        #print(type(predictions))
        print("predictions")
        print(predictions)
        _, predicted = torch.max(predictions, 1)
        #print(_)
        #print(predicted)
        #correct = (predicted == labels).sum()
        #accuracy = 100 * correct / total
        #print(accuracy)
        intlist = predicted.tolist()
        #probs = torch.softmax(predictions, dim=1)
        #winners = probs.argmax(dim=1)
        #corrects = (winners == target)
        #clf = np.where(predictions < 0.5, 0, 1)
        #print(clf)
        #for prediction in clf:
        #    intlist.append(int(prediction[0]))
        #    problist.append(int(prediction[0]))
        #shutil.rmtree("/tmp/tf" + str(myobj.modelInt) + myobj.period + myobj.mapname + str(count))
        del predictions
        del model
        return intlist, problist

    def do_test(self, request):
        np.random.seed(0)
        torch.manual_seed(0)
        dt = datetime.now()
        timestamp = dt.timestamp()
        myobj = json.loads(request.get_data(as_text=True), object_hook=lt.LearnTest)
        (config, model) = self.getmodel(myobj)
        if hasattr(myobj, 'testarray') and hasattr(myobj, 'testcatarray'):
            test = np.array(myobj.testarray, dtype='f')
            testcat = np.array(myobj.testcatarray, dtype='i')
        else:
            test = np.array(myobj.trainarray, dtype='f')
            testcat = np.array(myobj.traincatarray, dtype='i')
        global dicttask
        task = dicttask[myobj.mapname]
        accuracy_score = self.do_testinner(myobj, model, test, testcat, task)
        dt = datetime.now()
        print ("millis ", (dt.timestamp() - timestamp)*1000)
        return Response(json.dumps({"accuracy": float(accuracy_score)}), mimetype='application/json')

    def do_learntest(self, request):
        np.random.seed(0)
        torch.manual_seed(0)
        dt = datetime.now()
        timestamp = dt.timestamp()
        #print(request.get_data(as_text=True))
        myobj = json.loads(request.get_data(as_text=True), object_hook=lt.LearnTest)
        (config, model) = self.getmodel(myobj)
        accuracy_score = self.do_learntestinner(myobj, model, config)
        dt = datetime.now()
        print ("millis ", (dt.timestamp() - timestamp)*1000)
        return Response(json.dumps({"accuracy": float(accuracy_score)}), mimetype='application/json')

    def mytrain(self, model, inputs, labels, config, task):
        #print(labels)
        #print(labels)
        for i in range(config.steps):
            model.train()
            #print(type(inputs))
            #print(inputs)
            #print(type(labels))
            #print(labels)
            #exit
            #print(inputs.size())
            #print(labels.size())
            model.observe(inputs, task, labels)
        
    def mytrain2(self, model, inputs, labels, config, task):
        labels = labels.float()
        labels = labels.reshape(1, -1)
        labels = labels.reshape(-1, 1)
        #print(labels)
        criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #print("model parameters")
        #for param in model.parameters():
        #    print(param)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for i in range(config.steps):
            running_loss = 0.0
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, task)
            #print("ooo")
            #print(inputs)
            #print(outputs)
            #print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

    def zero(self, myobj):
        return hasattr(myobj, 'zero') or myobj.zero == False
            
    def do_learntestinner(self, myobj, model, config):
        array = np.array(myobj.trainingarray, dtype='f')
        cat = np.array(myobj.trainingcatarray, dtype='i')
        # NOTE class range 1 - 4 will be changed to 0 - 3
        # to avoid risk of being classified as 0 later
        if not self.zero(myobj):
            cat = cat - 1
        #print(len(cat))
        #print(array)
        #print(cat)
        if hasattr(myobj, 'testarray') and hasattr(myobj, 'testcatarray'):
            test = np.array(myobj.testarray, dtype='f')
            testcat = np.array(myobj.testcatarray, dtype='i')
            train = array
            traincat = cat
            # NOTE class range 1 - 4 will be changed to 0 - 3
            # to avoid risk of being classified as 0 later
            if not self.zero(myobj):
                testcat = testcat - 1
        else:
            (lenrow, lencol) = array.shape
            half = round(lenrow / 2)
            train = array[:half, :]
            test = array[half:, :]
            traincat = cat[:half]
            testcat = cat[half:]
            if len(cat) == 1:
                train = array
                test = array
                traincat = cat
                testcat = cat
        print("outcomes")
        print(myobj.outcomes)
        #print("cwd")
        #print(os.getcwd())
        #np.random.seed(0)
        #torch.manual_seed(0)
        #X, Y = make_moons(500, noise=0.2)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)
        #net = model.network(train, traincat)
        print(model)

        #net.train()

        v_x = torch.FloatTensor(train)
        v_y = torch.FloatTensor(traincat)
        #.reshape(1, -1)
        v_y = torch.LongTensor(traincat)
        #.reshape(1, -1)
        #print("tt")
        #print(traincat)
        #print(torch.FloatTensor(traincat).reshape(-1, 1))
        #print(torch.FloatTensor(traincat).view(-1, 1))
        #print(torch.FloatTensor(traincat).reshape(1, -1))
        #print(torch.FloatTensor(traincat).view(1, -1))
        #print(v_x)
        #print(v_y)
        global dicttask
        task = dicttask[myobj.mapname]
        if myobj.modelInt == 5:
            #print("bl", task, model.memory_data.size())
            mysize = model.memory_data.size()
            if task >= mysize[0]:
                model.memory_data = torch.cat([model.memory_data, torch.zeros(10, mysize[1], mysize[2])], 0)
                mysize = model.memory_data.size()
                #print("mysize", mysize)
                #mysize = model.memory_labs.size()
                #print("mysize2", mysize)
                #atens = torch.zeros(10, mysize[1])
                #print(atens.size(), atens)
                #atens = atens.long()
                #print(atens.size(), atens)
                #print(model.memory_labs)
                model.memory_labs = torch.cat([model.memory_labs, torch.zeros(10, mysize[1]).long()])
                mysize = model.grads.size()
                #print("mysize3", mysize)
                model.grads = torch.cat([model.grads, torch.zeros(mysize[0], 10)], 1)
                #print("mysize4", model.grads.size())
        if myobj.modelInt == 6:
            print("not implemented yet")
        self.mytrain(model, v_x, v_y, config, task)
        print("Trained task", task)
        if not self.taskone(myobj) and myobj.modelInt == 5:
            dicttask[myobj.mapname] = task + 1
        
        model.eval()
        if not (hasattr(myobj, 'save') and myobj.save == False):
            torch.save({'model': model, 'task': dicttask[myobj.mapname] }, self.getpath(myobj) + myobj.mapname + ".pt")
            #torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'task': dicttask[myobj.mapname] }, self.getpath(myobj) + myobj.mapname + ".pt")
        return self.do_testinner(myobj, model, test, testcat, task)

    def do_testinner(self, myobj, model, test, testcat, task):
        test_loss = 0
        accuracy_score = 0

        tv_x = torch.FloatTensor(test)
        tv_y = torch.LongTensor(testcat)
        
        y_hat = model(tv_x, task)
        #print("yhat", y_hat.size(), y_hat)
        (max_vals, arg_maxs) = torch.max(y_hat.data, dim=1) 
        # arg_maxs is tensor of indices [0, 1, 0, 2, 1, 1 . . ]
        num_correct = torch.sum(tv_y==arg_maxs)
        acc = float(num_correct) / len(testcat)
        #print(len(testcat))
        #print(tv_y)
        #print(arg_maxs)
        #print(max_vals)
        #print(y_hat.data)
        #print(tv_x)
        #y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
        #accuracy = np.sum(tv_y.reshape(-1,1) == y_hat_class) / len(testcat)
        accuracy_score = acc
        #train_loss.append(loss.item())

        print("testlen", len(testcat))
        print("test_loss")
        print(test_loss)
        print(accuracy_score)
        print(type(accuracy_score))
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
        return accuracy_score

    def getpath(self, myobj):
        if hasattr(myobj, 'path'):
            return myobj.path + '/'
        return '/tmp/'
    
    def taskone(self, myobj):
        return hasattr(myobj, 'taskone') and myobj.taskone == True

    def getmodel(self, myobj):
         (config, modelname) = self.getModel(myobj)
         global dictclass
         #print(dictclass.keys())
         if myobj.mapname in dictclass:
             model = dictclass[myobj.mapname]
         else:
            if os.path.isfile(self.getpath(myobj) + myobj.mapname + ".pt"):
                checkpoint = torch.load(self.getpath(myobj) + myobj.mapname + ".pt")
                model = checkpoint['model']
                task = checkpoint['task']
                model.eval()
                if hasattr(model, "memory_data"):
                    print("memorydata")
                    print(model.memory_data.size())
                    #print(model.memory_data[0])
                    #print(model.memory_data[1])
                    #print(model.memory_data[9])
                    #print(model.memory_data[10])
            else:
                Model = importlib.import_module('fb.model.' + modelname)
                #model = Model.Net(myobj, config)
                if not self.taskone(myobj) and myobj.modelInt == 5:
                    n_tasks = 10
                else:
                    n_tasks = initial_tasks
                model = Model.Net(myobj.size, myobj.outcomes, n_tasks, config)
                task = 0
            dictclass[myobj.mapname] = model
            global dicttask
            dicttask[myobj.mapname] = task
         return config, model
         
    def getModel(self, myobj):
        if myobj.modelInt == 1:
            modelname = 'single'
            config = myobj.gemSConfig
        if myobj.modelInt == 2:
            modelname = 'independent'
            config = myobj.gemIConfig
        if myobj.modelInt == 3:
            modelname = 'multimodal'
            config = myobj.gemMConfig
        if myobj.modelInt == 4:
            modelname = 'ewc'
            config = myobj.gemEWCConfig
        if myobj.modelInt == 5:
            modelname = 'gem'
            config = myobj.gemGEMConfig
        if myobj.modelInt == 6:
            modelname = 'icarl'
            config = myobj.gemiCaRLConfig
        return config, modelname;
    
    def do_learntestclassify(self, queue, request):
        dt = datetime.now()
        timestamp = dt.timestamp()
        myobj = json.loads(request.get_data(as_text=True), object_hook=lt.LearnTest)
        (config, model) = self.getmodel(myobj)
        accuracy_score = self.do_learntestinner(myobj, model, config)
        (intlist, problist) = self.do_classifyinner(myobj, model)
        print(len(intlist))
        print(intlist)
        print(problist)
        dt = datetime.now()
        print ("millis ", (dt.timestamp() - timestamp)*1000)
        queue.put(Response(json.dumps({"classifycatarray": intlist, "classifyprobarray": problist, "accuracy": float(accuracy_score)}), mimetype='application/json'))
