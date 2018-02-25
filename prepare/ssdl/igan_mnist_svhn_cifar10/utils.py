from six.moves import cPickle
import lasagne.layers as LL

def save(filename, param_values):
    with open(filename, 'wb') as f:
    	cPickle.dump(param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
    	param_values = cPickle.load(f)
    	return param_values

def save_model(filename, model):
	save(filename, LL.get_all_param_values(model))

def load_model(filename, model):
	param_values = load(filename)
	LL.set_all_param_values(model, param_values)
