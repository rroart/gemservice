#!/usr/bin/python3

from flask import Flask, request

import sys
from multiprocessing import Process, Queue

import json
from werkzeug.wrappers import Response

global cl

def classifyrunner2(queue, request):
    import classify
    cl = classify.Classify()
    cl.do_learntestclassify(queue, request)

def predictrunner(queue, request):
    import predict
    pr = predict.Predict()
    pr.do_learntestlist(queue, request)

def hasgpu():
    import torch
    return torch.cuda.is_available()
    
app = Flask(__name__)

@app.route('/eval', methods=['POST'])
def do_eval():
    return cl.do_eval(request)

@app.route('/classify', methods=['POST'])
def do_classify():
    return cl.do_classify(request)

@app.route('/test', methods=['POST'])
def do_test():
    return cl.do_test(request)

@app.route('/learntest', methods=['POST'])
def do_learntest():
    return cl.do_learntest(request)

@app.route('/learntestclassify', methods=['POST'])
def do_learntestclassify():
    def classifyrunner(queue, request):
        cl.do_learntestclassify(queue, request)
    queue = Queue()
    process = Process(target=classifyrunner, args=(queue, request))
    process.start()
    process.join()
    result = queue.get()
    return result

@app.route('/predictone', methods=['POST'])
def do_learntestpredictone():
    queue = Queue()
    process = Process(target=predictrunner, args=(queue, request))
    process.start()
    process.join()
    result = queue.get()
    return result

@app.route('/predict', methods=['POST'])
def do_learntestpredict():
    queue = Queue()
    process = Process(target=predictrunner, args=(queue, request))
    process.start()
    process.join()
    result = queue.get()
    return result

@app.route('/dataset', methods=['POST'])
def do_dataset():
    def classifyrunner(queue, request):
        try:
            cl.do_dataset(queue, request)
        except:
            queue.put(Response(json.dumps({"accuracy": None, "loss": None}), mimetype='application/json'))
            import sys,traceback
            traceback.print_exc(file=sys.stdout)
            print("\n")
            import random
            f = open("/tmp/outpt" + argstr() + str(random.randint(1000,9999)) + ".txt", "w")
            f.write(request.get_data(as_text=True))
            traceback.print_exc(file=f)
            f.close()
    queue = Queue()
    process = Process(target=classifyrunner, args=(queue, request))
    process.start()
    process.join()
    result = queue.get()
    return result

@app.route('/filename', methods=['POST'])
def do_filename():
    def filenamerunner(queue, request):
        import classify
        cl = classify.Classify()
        cl.do_filename(queue, request)
    queue = Queue()
    process = Process(target=filenamerunner, args=(queue, request))
    process.start()
    process.join()
    result = queue.get()
    return result

def argstr():
    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        return 'dev'
    else:
        return ''

if __name__ == '__main__':
#    queue = Queue()
#    process = Process(target=hasgpurunner, args=(queue, None))
#    process.start()
#    process.join()
#    hasgpu = queue.get()
    import classify
    global cl
    cl = classify.Classify()
    hasgpu = hasgpu()
    print("Has GPU", hasgpu)
    threaded = False
    if len(sys.argv) > 1 and (not hasgpu) and sys.argv[1] == 'multi':
        threaded = True
        print("Run threaded")
    port = 8040
    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        port = 8048
        print("Run other port")
    app.run(host='0.0.0.0', port=port, threaded=threaded)
