def culculate(file_path, inputs):
    import torch
    import numpy

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(file_path).to(my_device)
    input = torch.Tensor(inputs).to(my_device)
    outputs = model(input).detach().to("cpu").numpy().tolist()
    return outputs
