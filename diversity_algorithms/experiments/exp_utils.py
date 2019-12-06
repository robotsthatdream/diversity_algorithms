import sys
import getopt

#__all__={"RunParam", "analyse_params", "get_simple_params_dict"}

class RunParam(object):
    def __init__(self, short_name, default_value, doc):
        if (len(short_name)>1):
            print("Short parameter names should have only one character ! Value given: "+str(short_name))
            sys.exit(1)
        # some params may have an empty short_name: ""
        self.short_name=short_name
        self.default_value=default_value
        self.doc=doc
        self.value=None

    def set_value(self, value):
        try:
            self.value=type(self.default_value)(value)
        except ValueError:
            print("ERROR: the provided value for the argument is inappropriate. Value="+str(value)+", expected type="+str(type(self.default_value)))

    def get_value(self):
        if (self.value is not None):
            return self.value
        else:
            return self.default_value

    def is_default(self):
        return self.value==None

def check_params(params):
	short_params=["h"]
	OK=True
	for k in params.keys():
		if (params[k].short_name!="") and (params[k].short_name in short_params):
			print("ERROR: params are invalid, "+params[k].short_name+" appears several times !")
			OK=False
		short_params.append(params[k].short_name)
	if (not OK):
		sys.exit(1)

def get_param_from_short_name(params, short_name):
    for k in params.keys():
        if (short_name=="-"+params[k].short_name):
            return k
    return None

def get_simple_params_dict(params):
    dparams={}
    for k in params.keys():
        dparams[k]=params[k].get_value()
    return dparams

def analyze_params(params, argv):

    check_params(params)

    optstr="hv"+"".join([params[k].short_name+":" for k in params.keys()])
    optargs=[k+"=" for k in params.keys()]
    helpstr="-h -v ".join(["-"+params[k].short_name+" <"+k+">" for k in params.keys()])

    try:
        opts, args = getopt.getopt(argv[1:],optstr, optargs)
    except getopt.GetoptError:
        print(argv[0]+helpstr)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0]+helpstr)
            for k in params.keys():
                print("\t-"+params[k].short_name+" --"+k+": (default value: ("+params[k]+") "+params[k].doc)
            sys.exit(1)
        k=get_param_from_short_name(params, opt)
        if (k==None):
            if (opt[2:] in params.keys()):
                k=opt[2:]
            else:
                print("ERROR: unknown param: "+opt)
                sys.exit(1)
        if (k not in params.keys()):
            print("ERROR: non valid params: "+str(k))
            sys.exit(1)
        params[k].set_value(arg)
