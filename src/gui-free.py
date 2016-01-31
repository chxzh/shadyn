from shadow_optim import Optimizer, Renderer
def main():
    renderer = Renderer()
    renderer.start()
    optimizer = vanilla_optimizer(renderer)
    optimizer.start()

def vanilla_optimizer(renderer):
    optimizer = Optimizer()
    optimizer.set_method("Powell")
    

    return optimizer

if __name__ == "__main__":
    main()
    
