from shadow_optim import Optimizer
def schedule(*optimizers):
    # presuming they are all well-set
    for optimizer in optimizers:
        # presuming using the same renderer, previous result is fed as starting
        #     input params
        optimizer.run()