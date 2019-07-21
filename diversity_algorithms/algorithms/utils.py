import datetime
import os

def generate_exp_name(name=""):
    d=datetime.datetime.today()
    run_name=d.strftime(name+"%Y_%m_%d-%H:%M:%S")
    nb=0
    not_created=True
    while(not_created):
        try:
            os.mkdir(run_name+"_%d"%nb)
            run_name+="_%d"%nb
            not_created=False
        except OSError:
            nb+=1
    return run_name

