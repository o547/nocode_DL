def csv_out(file_path, model_path, input_size, output_size):
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torch import utils
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #ファイルの読み込み
    character_encoding = "utf-8"
    try:
        df = pd.read_csv(file_path, header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, header=0, encoding="shift_jis")
        character_encoding = "shift_jis"

    df = df.fillna(0)
    all_datas = df.iloc[0:, 0:input_size].to_numpy().astype(np.float32)

    datas = torch.from_numpy(all_datas).to(my_device)
    model = torch.jit.load(model_path).to(my_device)

    table = df.columns.to_numpy()
    while len(table) != input_size + output_size:
        table = np.append(table, "")
    table = table.reshape(1, input_size + output_size)  # (1,5) に変換

    #計算、書き込み
    for i in range(datas.size()[0]):
        in_data = datas[i].to("cpu").numpy()
        out = model(datas[i]).to("cpu").detach().numpy()
        column = np.concatenate([in_data, out])
        column = column.reshape(1, -1)  # (1,5) に変換
        table = np.append(table, column, axis=0)
    df_table = pd.DataFrame(table)
    df_table.to_csv(
        file_path.replace(".csv", "_out.csv"),
        header=False,
        index=False,
        encoding=character_encoding,
    )
